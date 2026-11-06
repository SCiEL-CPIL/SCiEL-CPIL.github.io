import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    for c in range(n + 1, n + 1000):
        if _is_prime(c):
            return c
    return n + 1

def _mod_inverse(a, p):
    """Extended Euclidean algorithm for modular inverse."""
    if a < 0:
        a += p
    g, x, _ = _egcd(a, p)
    if g != 1:
        return None
    return x % p

def _egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def _build_sidon_mod_p(p, max_size):
    """
    Build a Sidon set modulo p using quadratic residues.
    For prime p ≡ 3 (mod 4), the set {x^2 mod p : x in [1, (p-1)/2]} is a Sidon set.
    Actually, we need to be more careful. Let's use a known construction:
    For p prime, the set S = {x : x^2 = y mod p for some y} has special properties.
    
    A better approach: Use the Singer construction or simply greedily build
    a Sidon set modulo p.
    """
    # Greedy construction of Sidon set modulo p
    S = []
    sums = set()
    diffs = set()
    
    for x in range(1, p):
        valid = True
        for y in S:
            s1 = (x + y) % p
            s2 = (x + x) % p
            if s1 in sums or s2 in sums:
                valid = False
                break
            d = abs(x - y)
            d = min(d, p - d)  # Take the smaller difference
            if d in diffs:
                valid = False
                break
        if valid:
            S.append(x)
            sums.add((x + x) % p)
            for y in S[:-1]:
                sums.add((x + y) % p)
                d = abs(x - y)
                d = min(d, p - d)
                diffs.add(d)
        if len(S) >= max_size:
            break
    
    return S

def _try_construct_with_prime(p, target_size_each):
    """
    Try to construct two Sidon sets A and B in [1, N] using modular arithmetic.
    We pick a prime p > 2*N, build Sidon sets modulo p, then map to [1, N].
    """
    if p <= 2 * N:
        return None, None
    
    # Build Sidon set modulo p
    S1 = _build_sidon_mod_p(p, target_size_each)
    if len(S1) < 2:
        return None, None
    
    # Now we need to map these to [1, N] while preserving Sidon property
    # and ensuring disjoint difference sets.
    # 
    # Strategy: Use a linear map x -> a*x + b mod p, then take values in [1, N].
    # But this is tricky. Instead, let's use a different approach:
    # 
    # Direct greedy construction in [1, N] with the constraint that
    # differences are disjoint. We use the modular structure to guide
    # the choice of elements.
    
    # Actually, let's just do a smart greedy with bitset optimization.
    # The key insight from the inspirations is to use bitsets for fast
    # collision detection.
    
    return None, None

def _can_add_bitset(x, curr_set, curr_sums, curr_diffs, other_diffs):
    """Check if x can be added to curr_set using precomputed bitsets."""
    # Check self-sum
    if curr_sums[x + x]:
        return False
    
    # Check all pairs with existing elements
    for y in curr_set:
        if curr_sums[x + y]:
            return False
        d = abs(x - y)
        if curr_diffs[d] or other_diffs[d]:
            return False
    
    return True

def _greedy_build_with_bitsets(rng, target_size_each):
    """
    Greedy construction using bitsets for fast constraint checking.
    """
    A = []
    B = []
    
    # Bitsets for sums and differences
    # Sums range from 2 to 2*N = 512
    # Diffs range from 1 to N-1 = 255
    sum_A = [0] * (2 * N + 1)
    sum_B = [0] * (2 * N + 1)
    diff_A = [0] * (N + 1)
    diff_B = [0] * (N + 1)
    
    # Try to add elements 1..N to A and B
    for x in range(1, N + 1):
        # Try adding to A
        if len(A) < target_size_each:
            valid = True
            # Check self-sum
            if sum_A[x + x]:
                valid = False
            else:
                for y in A:
                    if sum_A[x + y]:
                        valid = False
                        break
                    d = abs(x - y)
                    if diff_A[d] or diff_B[d]:
                        valid = False
                        break
            if valid:
                A.append(x)
                sum_A[x + x] = 1
                for y in A[:-1]:
                    sum_A[x + y] = 1
                    d = abs(x - y)
                    diff_A[d] = 1
                continue
        
        # Try adding to B
        if len(B) < target_size_each:
            valid = True
            if sum_B[x + x]:
                valid = False
            else:
                for y in B:
                    if sum_B[x + y]:
                        valid = False
                        break
                    d = abs(x - y)
                    if diff_B[d] or diff_A[d]:
                        valid = False
                        break
            if valid:
                B.append(x)
                sum_B[x + x] = 1
                for y in B[:-1]:
                    sum_B[x + y] = 1
                    d = abs(x - y)
                    diff_B[d] = 1
    
    return A, B, sum_A, sum_B, diff_A, diff_B

def _local_search_bitset(A, B, sum_A, sum_B, diff_A, diff_B, rng, max_time):
    """
    Local search using bitsets for fast constraint checking.
    """
    current_size = len(A) + len(B)
    best_size = current_size
    best_A = list(A)
    best_B = list(B)
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() < max_time:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try Add
            target = rng.choice([0, 1])
            curr_set = A if target == 0 else B
            if len(curr_set) < MAX_EACH:
                # Pick a random candidate not in A or B
                all_elements = set(A) | set(B)
                candidates = [x for x in range(1, N + 1) if x not in all_elements]
                if candidates:
                    x = rng.choice(candidates)
                    
                    if target == 0:
                        valid = True
                        if sum_A[x + x]:
                            valid = False
                        else:
                            for y in A:
                                if sum_A[x + y]:
                                    valid = False
                                    break
                                d = abs(x - y)
                                if diff_A[d] or diff_B[d]:
                                    valid = False
                                    break
                        if valid:
                            A.append(x)
                            sum_A[x + x] = 1
                            for y in A[:-1]:
                                sum_A[x + y] = 1
                                d = abs(x - y)
                                diff_A[d] = 1
                            new_size = len(A) + len(B)
                            if new_size > best_size:
                                best_size = new_size
                                best_A = list(A)
                                best_B = list(B)
                            T *= cooling
                            continue
                    else:
                        valid = True
                        if sum_B[x + x]:
                            valid = False
                        else:
                            for y in B:
                                if sum_B[x + y]:
                                    valid = False
                                    break
                                d = abs(x - y)
                                if diff_B[d] or diff_A[d]:
                                    valid = False
                                    break
                        if valid:
                            B.append(x)
                            sum_B[x + x] = 1
                            for y in B[:-1]:
                                sum_B[x + y] = 1
                                d = abs(x - y)
                                diff_B[d] = 1
                            new_size = len(A) + len(B)
                            if new_size > best_size:
                                best_size = new_size
                                best_A = list(A)
                                best_B = list(B)
                            T *= cooling
                            continue
        elif move_type < 0.6:
            # Try Remove
            if len(A) > 0 and len(B) > 0:
                if len(A) >= len(B):
                    target = 0
                else:
                    target = 1
                
                curr_set = A if target == 0 else B
                if curr_set:
                    x = rng.choice(curr_set)
                    
                    # Remove x
                    if target == 0:
                        idx = A.index(x)
                        A.pop(idx)
                        # Rebuild sums and diffs for A
                        new_sum_A = [0] * (2 * N + 1)
                        new_diff_A = [0] * (N + 1)
                        for i in range(len(A)):
                            for j in range(i, len(A)):
                                new_sum_A[A[i] + A[j]] = 1
                                if i != j:
                                    d = abs(A[i] - A[j])
                                    new_diff_A[d] = 1
                        sum_A[:] = new_sum_A
                        diff_A[:] = new_diff_A
                    else:
                        idx = B.index(x)
                        B.pop(idx)
                        new_sum_B = [0] * (2 * N + 1)
                        new_diff_B = [0] * (N + 1)
                        for i in range(len(B)):
                            for j in range(i, len(B)):
                                new_sum_B[B[i] + B[j]] = 1
                                if i != j:
                                    d = abs(B[i] - B[j])
                                    new_diff_B[d] = 1
                        sum_B[:] = new_sum_B
                        diff_B[:] = new_diff_B
                    
                    new_size = len(A) + len(B)
                    delta = new_size - current_size
                    if rng.random() < pow(2.71828, delta / T):
                        current_size = new_size
                        T *= cooling
                    else:
                        # Revert
                        if target == 0:
                            A.append(x)
                            new_sum_A = [0] * (2 * N + 1)
                            new_diff_A = [0] * (N + 1)
                            for i in range(len(A)):
                                for j in range(i, len(A)):
                                    new_sum_A[A[i] + A[j]] = 1
                                    if i != j:
                                        d = abs(A[i] - A[j])
                                        new_diff_A[d] = 1
                            sum_A[:] = new_sum_A
                            diff_A[:] = new_diff_A
                        else:
                            B.append(x)
                            new_sum_B = [0] * (2 * N + 1)
                            new_diff_B = [0] * (N + 1)
                            for i in range(len(B)):
                                for j in range(i, len(B)):
                                    new_sum_B[B[i] + B[j]] = 1
                                    if i != j:
                                        d = abs(B[i] - B[j])
                                        new_diff_B[d] = 1
                            sum_B[:] = new_sum_B
                            diff_B[:] = new_diff_B
        else:
            # Try Swap: Remove x from one set, add y to other
            if len(A) > 0 and len(B) < MAX_EACH:
                x = rng.choice(A)
                # Tentatively remove x from A
                idx = A.index(x)
                A.pop(idx)
                new_sum_A = [0] * (2 * N + 1)
                new_diff_A = [0] * (N + 1)
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        new_sum_A[A[i] + A[j]] = 1
                        if i != j:
                            d = abs(A[i] - A[j])
                            new_diff_A[d] = 1
                sum_A[:] = new_sum_A
                diff_A[:] = new_diff_A
                
                # Try to add y to B
                all_elements = set(A) | set(B)
                candidates = [y for y in range(1, N + 1) if y not in all_elements]
                found = False
                for _ in range(5):
                    if not candidates:
                        break
                    y = rng.choice(candidates)
                    valid = True
                    if sum_B[y + y]:
                        valid = False
                    else:
                        for z in B:
                            if sum_B[y + z]:
                                valid = False
                                break
                            d = abs(y - z)
                            if diff_B[d] or diff_A[d]:
                                valid = False
                                break
                    if valid:
                        B.append(y)
                        sum_B[y + y] = 1
                        for z in B[:-1]:
                            sum_B[y + z] = 1
                            d = abs(y - z)
                            diff_B[d] = 1
                        new_size = len(A) + len(B)
                        if new_size > best_size:
                            best_size = new_size
                            best_A = list(A)
                            best_B = list(B)
                        found = True
                        break
                
                if not found:
                    # Revert
                    A.append(x)
                    new_sum_A = [0] * (2 * N + 1)
                    new_diff_A = [0] * (N + 1)
                    for i in range(len(A)):
                        for j in range(i, len(A)):
                            new_sum_A[A[i] + A[j]] = 1
                            if i != j:
                                d = abs(A[i] - A[j])
                                new_diff_A[d] = 1
                    sum_A[:] = new_sum_A
                    diff_A[:] = new_diff_A
            
            elif len(B) > 0 and len(A) < MAX_EACH:
                x = rng.choice(B)
                idx = B.index(x)
                B.pop(idx)
                new_sum_B = [0] * (2 * N + 1)
                new_diff_B = [0] * (N + 1)
                for i in range(len(B)):
                    for j in range(i, len(B)):
                        new_sum_B[B[i] + B[j]] = 1
                        if i != j:
                            d = abs(B[i] - B[j])
                            new_diff_B[d] = 1
                sum_B[:] = new_sum_B
                diff_B[:] = new_diff_B
                
                all_elements = set(A) | set(B)
                candidates = [y for y in range(1, N + 1) if y not in all_elements]
                found = False
                for _ in range(5):
                    if not candidates:
                        break
                    y = rng.choice(candidates)
                    valid = True
                    if sum_A[y + y]:
                        valid = False
                    else:
                        for z in A:
                            if sum_A[y + z]:
                                valid = False
                                break
                            d = abs(y - z)
                            if diff_A[d] or diff_B[d]:
                                valid = False
                                break
                    if valid:
                        A.append(y)
                        sum_A[y + y] = 1
                        for z in A[:-1]:
                            sum_A[y + z] = 1
                            d = abs(y - z)
                            diff_A[d] = 1
                        new_size = len(A) + len(B)
                        if new_size > best_size:
                            best_size = new_size
                            best_A = list(A)
                            best_B = list(B)
                        found = True
                        break
                
                if not found:
                    B.append(x)
                    new_sum_B = [0] * (2 * N + 1)
                    new_diff_B = [0] * (N + 1)
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            new_sum_B[B[i] + B[j]] = 1
                            if i != j:
                                d = abs(B[i] - B[j])
                                new_diff_B[d] = 1
                    sum_B[:] = new_sum_B
                    diff_B[:] = new_diff_B
        
        if T < 0.01:
            break
    
    return best_A, best_B, best_size

def _polish(A, B, rng, max_time):
    """
    Final polish pass: try all possible single-element replacements.
    """
    sum_A = [0] * (2 * N + 1)
    sum_B = [0] * (2 * N + 1)
    diff_A = [0] * (N + 1)
    diff_B = [0] * (N + 1)
    
    for i in range(len(A)):
        for j in range(i, len(A)):
            sum_A[A[i] + A[j]] = 1
            if i != j:
                diff_A[abs(A[i] - A[j])] = 1
    for i in range(len(B)):
        for j in range(i, len(B)):
            sum_B[B[i] + B[j]] = 1
            if i != j:
                diff_B[abs(B[i] - B[j])] = 1
    
    best_size = len(A) + len(B)
    best_A = list(A)
    best_B = list(B)
    
    # Try replacing each element in A with a better one
    for i in range(len(A)):
        old = A[i]
        A.remove(old)
        
        # Rebuild A sums/diffs
        new_sum_A = [0] * (2 * N + 1)
        new_diff_A = [0] * (N + 1)
        for k in range(len(A)):
            for l in range(k, len(A)):
                new_sum_A[A[k] + A[l]] = 1
                if k != l:
                    new_diff_A[abs(A[k] - A[l])] = 1
        sum_A[:] = new_sum_A
        diff_A[:] = new_diff_A
        
        # Try to add a new element
        all_elements = set(A) | set(B)
        candidates = [x for x in range(1, N + 1) if x not in all_elements]
        found = False
        for x in candidates:
            valid = True
            if sum_A[x + x]:
                valid = False
            else:
                for y in A:
                    if sum_A[x + y]:
                        valid = False
                        break
                    d = abs(x - y)
                    if diff_A[d] or diff_B[d]:
                        valid = False
                        break
            if valid:
                A.append(x)
                sum_A[x + x] = 1
                for y in A[:-1]:
                    sum_A[x + y] = 1
                    d = abs(x - y)
                    diff_A[d] = 1
                found = True
                break
        
        if not found:
            A.append(old)
            new_sum_A = [0] * (2 * N + 1)
            new_diff_A = [0] * (N + 1)
            for k in range(len(A)):
                for l in range(k, len(A)):
                    new_sum_A[A[k] + A[l]] = 1
                    if k != l:
                        new_diff_A[abs(A[k] - A[l])] = 1
            sum_A[:] = new_sum_A
            diff_A[:] = new_diff_A
        
        new_size = len(A) + len(B)
        if new_size > best_size:
            best_size = new_size
            best_A = list(A)
            best_B = list(B)
    
    # Try replacing each element in B with a better one
    for i in range(len(B)):
        old = B[i]
        B.remove(old)
        
        new_sum_B = [0] * (2 * N + 1)
        new_diff_B = [0] * (N + 1)
        for k in range(len(B)):
            for l in range(k, len(B)):
                new_sum_B[B[k] + B[l]] = 1
                if k != l:
                    new_diff_B[abs(B[k] - B[l])] = 1
        sum_B[:] = new_sum_B
        diff_B[:] = new_diff_B
        
        all_elements = set(A) | set(B)
        candidates = [x for x in range(1, N + 1) if x not in all_elements]
        found = False
        for x in candidates:
            valid = True
            if sum_B[x + x]:
                valid = False
            else:
                for y in B:
                    if sum_B[x + y]:
                        valid = False
                        break
                    d = abs(x - y)
                    if diff_B[d] or diff_A[d]:
                        valid = False
                        break
            if valid:
                B.append(x)
                sum_B[x + x] = 1
                for y in B[:-1]:
                    sum_B[x + y] = 1
                    d = abs(x - y)
                    diff_B[d] = 1
                found = True
                break
        
        if not found:
            B.append(old)
            new_sum_B = [0] * (2 * N + 1)
            new_diff_B = [0] * (N + 1)
            for k in range(len(B)):
                for l in range(k, len(B)):
                    new_sum_B[B[k] + B[l]] = 1
                    if k != l:
                        new_diff_B[abs(B[k] - B[l])] = 1
            sum_B[:] = new_sum_B
            diff_B[:] = new_diff_B
        
        new_size = len(A) + len(B)
        if new_size > best_size:
            best_size = new_size
            best_A = list(A)
            best_B = list(B)
    
    return best_A, best_B, best_size

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_size = 0
    best_A = []
    best_B = []
    
    rng = random.Random(42)
    
    # Run multiple restarts with different random seeds
    for seed in range(100):
        if time.time() > deadline - 10.0:
            break
        
        rng.seed(seed)
        A, B, sum_A, sum_B, diff_A, diff_B = _greedy_build_with_bitsets(rng, MAX_EACH)
        
        if len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A = list(A)
            best_B = list(B)
        
        # Run local search
        sa_deadline = min(time.time() + 3.0, deadline - 5.0)
        A, B, size = _local_search_bitset(A, B, sum_A, sum_B, diff_A, diff_B, rng, sa_deadline)
        
        if size > best_size:
            best_size = size
            best_A = list(A)
            best_B = list(B)
        
        # Run polish
        if time.time() < deadline - 3.0:
            A, B, size = _polish(A, B, rng, deadline - 2.0)
            if size > best_size:
                best_size = size
                best_A = list(A)
                best_B = list(B)
        
        if best_size >= 2 * MAX_EACH:
            break
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]

if __name__ == "__main__":
    print(construct())
