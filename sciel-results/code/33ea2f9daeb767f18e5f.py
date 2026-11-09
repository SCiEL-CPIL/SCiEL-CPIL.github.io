"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Try to find a lower rank decomposition using structured random search
    # The tensor for matrix multiplication has a special structure:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition requires: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all (i,j,k)
    
    # For a 2x5x6 matrix multiplication, known lower bounds suggest rank could be lower than 60
    # Let's try a structured approach with rank reduction
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    
    # Try various ranks below 60 using random initialization + local search
    # The key constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # And for "wrong" combinations (where the triple doesn't correspond to a valid multiplication term), it should be 0
    
    # Actually, the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because every (i,j,k) triple contributes to C[i,k] via A[i,j]*B[j,k]
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k
    
    # This is a rank-1 decomposition of the all-ones tensor of shape (n*m, m*p, p*n)
    # Wait, that's not right. Let me re-read the convention.
    
    # W[k*n+i, r] - so W is indexed by (k, i) where k is column of B / row of C, i is row of A / row of C
    # The decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k] = C[i,k]
    # This must hold for all A, B, so:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (identity-like)
    
    # More precisely, for each fixed (i,k), we need:
    # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k] = C[i,k]
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # There are no other constraints because the tensor is just the all-ones tensor over (i,j,k) indices
    # mapped to the flattened indices.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,2), j in [0,5), k in [0,6)
    
    # This is finding a rank-r decomposition of the 10x30x12 all-ones tensor!
    # The all-ones tensor of shape (a,b,c) has rank 1 (just all ones vectors).
    # But wait - the indices are coupled: U depends on (i,j), V on (j,k), W on (k,i)
    # The coupling through j and k and i means it's NOT just an all-ones tensor.
    
    # Let me think again. We need for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # The left side couples all three indices. Let's define:
    # For a fixed r, let u_r be a vector of length n*m, v_r of length m*p, w_r of length p*n
    # Then we need: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
    
    # This is a constrained tensor decomposition where the tensor is all-ones but the
    # index structure creates coupling.
    
    # One trivial solution: rank 1 with u_r[i*m+j]=1, v_r[j*p+k]=1, w_r[k*n+i]=1
    # Check: sum over r=1: 1*1*1 = 1. Yes! Rank 1 works!
    
    # Wait, that can't be right. Let me verify with the actual matrix multiplication.
    # If U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1:
    # Then the computed "C[i,k]" = sum_j sum_r A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # = sum_j A[i,j]*B[j,k]*1*1*1 = sum_j A[i,j]*B[j,k]
    # That's exactly matrix multiplication!
    
    # So rank 1 is sufficient?? That seems too good to be true. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 6 matrix"
    # The tensor here is the matrix multiplication tensor, which has a specific structure.
    # The standard matrix multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A times 
    # the (j,k) entry of B contributes to C[i,k]. 
    
    # In the standard formulation, the tensor is NOT all-ones. It's:
    # T_{(i,j),(j,k),(k,i)} = 1 (using the index conventions given)
    # And all other entries are 0.
    
    # But with the flattening: U index = i*m+j, V index = j*p+k, W index = k*n+i
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # And T[a, b, c] = 0 otherwise (i.e., for index combinations that don't correspond to valid (i,j,k))
    
    # So it IS the all-ones tensor on the valid index set, and the valid index set is defined by
    # the coupling: U index (i,j), V index (j,k), W index (k,i) must be consistent.
    
    # For the all-ones vectors: u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    # The product u[i*m+j]*v[j*p+k]*w[k*n+i] = 1 for all valid (i,j,k)
    # But what about invalid combinations? The tensor only has entries at valid (i,j,k) positions.
    # The decomposition only needs to match on the valid entries.
    
    # So yes, rank 1 should work! Let me double-check by considering what "rank" means here.
    # In tensor decomposition, the rank of a tensor T of shape (I,J,K) is the minimum r such that
    # T[a,b,c] = sum_{r=1}^{R} U[a,r]*V[b,r]*W[c,r] for all a,b,c.
    
    # If our tensor is zero everywhere except at positions (i*m+j, j*p+k, k*n+i) where it's 1,
    # then we need the decomposition to give 1 at those positions and 0 at all other positions.
    
    # With all-ones vectors, we get 1 at ALL positions (a,b,c), not just the valid ones.
    # So we need the decomposition to be 0 at invalid positions.
    
    # Invalid positions are those where the indices (a,b,c) don't correspond to a consistent (i,j,k).
    # For example, if a = i1*m+j1, b = j2*p+k1, c = k2*n+i2 where j1 != j2 or i1 != i2 or k1 != k2.
    
    # So the all-ones rank-1 decomposition does NOT work because it gives 1 at invalid positions too.
    
    # We need a more careful analysis. The tensor is:
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 0 otherwise
    
    # The number of valid positions is n*m*p = 60.
    # The total number of positions is (n*m)*(m*p)*(p*n) = 10*30*12 = 3600.
    
    # So we need a tensor decomposition that is 1 on 60 specific positions and 0 on the other 3540.
    
    # This is a much harder problem. The schoolbook rank is 60 (one rank-1 term per valid position).
    # Can we do better?
    
    # Let's try a random search approach for lower ranks.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly matches the matrix multiplication tensor."""
        rank = len(U[0])
        # Check all valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        # Check some invalid positions (sample)
        # An invalid position: pick a,b,c that don't form a consistent (i,j,k)
        # For thoroughness, check all positions would be expensive, so sample
        for _ in range(1000):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if this is a valid position
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            if not (i1 == i2 and j1 == j2 and k1 == k2):
                # Invalid position, should be 0
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s) > 1e-10:
                    return False
        return True
    
    def verify_decomposition_full(U, V, W, n, m, p):
        """Full verification of all positions."""
        rank = len(U[0])
        # Check all valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(s - 1.0) > 1e-10:
                        return False
        # Check all invalid positions
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (i1 == i2 and j1 == j2 and k1 == k2):
                        s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                        if abs(s) > 1e-10:
                            return False
        return True
    
    # Try rank 1: all ones
    # Already established this doesn't work because of invalid positions
    
    # Let's try a structured approach. 
    # Key insight: The tensor has a specific algebraic structure.
    # For 2x5x6, let's see if there's a known low-rank decomposition.
    
    # Actually, let me reconsider. The problem says "integer/half-integer entries".
    # This suggests the answer might involve small rational numbers.
    
    # Let me try a different approach: use the fact that the tensor can be written as
    # a sum of rank-1 tensors where each rank-1 term is supported on a subset of valid positions.
    
    # For rank r, we need r rank-1 tensors whose sum equals the target tensor.
    # Each rank-1 tensor U_r * V_r * W_r has the property that its support is a "combinatorial rectangle"
    # in the (i,j,k) space (intersected with valid positions).
    
    # Let me try random search with gradient-free optimization for small ranks.
    
    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition."""
        rng = random.Random(seed)
        
        # Initialize with small random values
        U = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
        V = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
        W = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
        
        # Build list of valid and invalid positions
        valid_positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_positions.append((i*m+j, j*p+k, k*n+i, 1.0))
        
        invalid_positions = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (i1 == i2 and j1 == j2 and k1 == k2):
                        invalid_positions.append((a, b, c, 0.0))
        
        # Sample invalid positions for efficiency
        invalid_sample = rng.sample(invalid_positions, min(500, len(invalid_positions)))
        
        def compute_loss(U, V, W):
            loss = 0.0
            for a, b, c, target in valid_positions:
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(r))
                loss += (s - target) ** 2
            for a, b, c, target in invalid_sample:
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(r))
                loss += (s - target) ** 2
            return loss
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            loss = compute_loss(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if best_loss < 1e-12:
                break
            
            # Random perturbation (simulated annealing style)
            step_size = 0.5 * (1 - iteration / max_iter) + 0.1
            idx = rng.randint(0, 2)
            if idx == 0:
                row = rng.randint(0, n*m-1)
                col = rng.randint(0, r-1)
                U[row][col] += rng.gauss(0, step_size)
            elif idx == 1:
                row = rng.randint(0, m*p-1)
                col = rng.randint(0, r-1)
                V[row][col] += rng.gauss(0, step_size)
            else:
                row = rng.randint(0, p*n-1)
                col = rng.randint(0, r-1)
                W[row][col] += rng.gauss(0, step_size)
        
        if best_loss < 1e-8:
            # Verify with full check
            if verify_decomposition_full(best_U, best_V, best_W, n, m, p):
                return best_U, best_V, best_W
        return None
    
    # Try ranks from 2 upward (rank 1 is impossible as shown)
    for r in range(2, min(rank_schoolbook, 20)):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, max_iter=3000, seed=r*1000)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            print(f"Found rank {r} decomposition!")
            break
    
    # If no better rank found, try a more systematic approach for specific small ranks
    # Try rank 2, 3 with different initializations
    if best_rank == rank_schoolbook:
        for r in range(2, 8):
            if time.time() > deadline - 10:
                break
            for seed in range(5):
                result = try_rank(r, max_iter=5000, seed=seed*10000 + r)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = r
                    break
            if best_rank < rank_schoolbook:
                break
    
    # Final time check
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
