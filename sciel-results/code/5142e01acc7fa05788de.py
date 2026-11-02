"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 110.0  # safety margin within 120s


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k * n + i  # column index in [0, n*p)
            # U[i*m+j, c] = 1 for all j
            for j in range(m):
                U[i * m + j, c] = 1
            # V[j*p+k, c] = 1 for all j
            for j in range(m):
                V[j * p + k, c] = 1
            # W[k*n+i, c] = 1
            W[k * n + i, c] = 1
    return U, V, W


def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Faster vectorized check. U, V, W are 2D arrays (rows, rank)."""
    for i in range(n):
        for k in range(p):
            w = W[k * n + i]
            s = 0.0
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                if np.any(u != 0) and np.any(v != 0):
                    s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros(n * m * rank, dtype=int)
    V = np.zeros(m * p * rank, dtype=int)
    W = np.zeros(p * n * rank, dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i * m + j) * rank + c] = 1
                V[(j * p + k) * rank + c] = 1
                W[(k * n + i) * rank + c] = 1
                c += 1
    U = U.reshape(n * m, rank)
    V = V.reshape(m * p, rank)
    W = W.reshape(p * n, rank)
    return U, V, W


def _try_smaller_rank(target_rank, n, m, p, attempts=200):
    """
    For ranks < n*p, try to find a decomposition using integer entries.
    Uses structured W matrices and solves for U, V.
    """
    import random
    from itertools import product
    
    # Try different W structures: each column is assigned to one or more (i,k) pairs
    for _ in range(attempts):
        if time.time() > DEADLINE:
            return None
        
        # Generate W: each column gets 1 or 2 (i,k) assignments
        W = np.zeros((p * n, target_rank), dtype=float)
        for c in range(target_rank):
            # Decide how many (i,k) pairs this column serves
            num_assign = random.choices([1, 2], weights=[0.7, 0.3])[0]
            assigned = set()
            for _ in range(num_assign):
                k = random.randint(0, p - 1)
                i = random.randint(0, n - 1)
                if (i, k) not in assigned:
                    assigned.add((i, k))
                    W[k * n + i, c] = 1.0
        
        # Check every (i,k) has at least one column
        valid = True
        for i in range(n):
            for k in range(p):
                if W[k * n + i].sum() == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        
        # Now solve for U and V
        # For each (i,k), let S_{i,k} = {c : W[k*n+i,c] != 0}
        # We need: sum_{c in S_{i,k}} U[i*m+j,c] * V[j*p+k,c] = 1 for all j
        # Strategy: set U[i*m+j,c] = 1 for first c in S_{i,k}, 0 for others
        # Then V[j*p+k,c_first] = 1 for all j
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        
        solved = True
        for i in range(n):
            for k in range(p):
                S = np.where(W[k * n + i] != 0)[0]
                if len(S) == 0:
                    solved = False
                    break
                c_first = S[0]
                for j in range(m):
                    U[i * m + j, c_first] = 1.0
                    V[j * p + k, c_first] = 1.0
                # Zero out other columns for this (i,k) in U and V
                for c in S[1:]:
                    for j in range(m):
                        U[i * m + j, c] = 0.0
                        V[j * p + k, c] = 0.0
            if not solved:
                break
        
        if not solved:
            continue
        
        # Verify
        ok = _verify_np(U, V, W, n, m, p)
        if ok:
            return {'U': U, 'V': V, 'W': W}
    
    return None


def _try_kronecker_structure(n, m, p):
    """
    Try structured decompositions for rank < n*p.
    Key insight: for rank r < n*p, some columns must serve multiple (i,k) pairs.
    We try to find such structures by solving linear systems.
    """
    import random
    from itertools import combinations
    
    # For each rank from n*p-1 down, try to find a valid decomposition
    for rank in range(n * p - 1, max(n * p - 5, 5), -1):
        if time.time() > DEADLINE - 15:
            break
        
        # Try multiple random W structures
        for attempt in range(200):
            if time.time() > DEADLINE - 15:
                break
            
            # Generate W where each column is assigned to 1 or 2 (i,k) pairs
            W = np.zeros((p * n, rank), dtype=float)
            for c in range(rank):
                # 80% chance of single assignment, 20% of double
                if random.random() < 0.8:
                    k = random.randint(0, p - 1)
                    i = random.randint(0, n - 1)
                    W[k * n + i, c] = 1.0
                else:
                    # Assign to two different (i,k) pairs
                    k1 = random.randint(0, p - 1)
                    i1 = random.randint(0, n - 1)
                    k2 = random.randint(0, p - 1)
                    i2 = random.randint(0, n - 1)
                    if (i1, k1) != (i2, k2):
                        W[k1 * n + i1, c] = 1.0
                        W[k2 * n + i2, c] = 1.0
                    else:
                        W[k1 * n + i1, c] = 1.0
            
            # Check every (i,k) has coverage
            valid = True
            for i in range(n):
                for k in range(p):
                    if W[k * n + i].sum() == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Solve for U and V using the "first column" strategy
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            
            solved = True
            for i in range(n):
                for k in range(p):
                    S = np.where(W[k * n + i] != 0)[0]
                    if len(S) == 0:
                        solved = False
                        break
                    c_first = S[0]
                    for j in range(m):
                        U[i * m + j, c_first] = 1.0
                        V[j * p + k, c_first] = 1.0
                    for c in S[1:]:
                        for j in range(m):
                            U[i * m + j, c] = 0.0
                            V[j * p + k, c] = 0.0
                if not solved:
                    break
            
            if not solved:
                continue
            
            if _verify_np(U, V, W, n, m, p):
                return {'U': U, 'V': V, 'W': W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Best result found so far (start with schoolbook)
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p

    # 1) Try rank = n*p = 14
    U14, V14, W14 = _construct_np(n, m, p)
    if _verify_np(U14, V14, W14, n, m, p):
        best_U, best_V, best_W = U14, V14, W14
        best_rank = n * p
        print(f"Found rank {n*p} decomposition")
    else:
        print(f"Rank {n*p} construction failed verification")

    # 2) Try Kronecker structure
    if best_rank > 10:
        result = _try_kronecker_structure(n, m, p)
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = result['U'].shape[1]
            print(f"Found rank {best_rank} decomposition via Kronecker")

    # 3) Try ranks from n*p - 1 down to 1
    # Allocate more time to higher ranks (closer to n*p) as they're more likely to succeed
    time_remaining = DEADLINE - time.time() - 5  # 5s safety margin
    if time_remaining > 0:
        ranks_to_try = list(range(n * p - 1, 0, -1))
        # Weight attempts: more attempts for ranks close to n*p
        total_weight = sum(1.0 / (1 + abs(rank - n * p)) for rank in ranks_to_try)
        for rank in ranks_to_try:
            if time.time() > DEADLINE - 5:
                break
            weight = 1.0 / (1 + abs(rank - n * p))
            attempts = max(50, int(time_remaining * weight / total_weight * 10))
            attempts = min(attempts, 500)
            result = _try_smaller_rank(rank, n, m, p, attempts=attempts)
            if result is not None:
                best_U, best_V, best_W = result['U'], result['V'], result['W']
                best_rank = rank
                print(f"Found rank {rank} decomposition")
                break

    # Convert to list-of-lists with integer/half-integer entries
    def to_int_list(mat):
        # Multiply by 2 to handle half-integers, check if all even, then divide
        m2 = mat * 2
        if np.allclose(m2, np.round(m2), atol=1e-9):
            m2 = np.round(m2).astype(int)
            if np.all(m2 % 2 == 0):
                return (m2 // 2).tolist()
            return (m2.tolist())
        return mat.tolist()

    return {
        'U': to_int_list(best_U),
        'V': to_int_list(best_V),
        'W': to_int_list(best_W)
    }
