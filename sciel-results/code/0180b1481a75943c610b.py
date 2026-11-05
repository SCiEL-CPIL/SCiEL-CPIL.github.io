import random
import time
import numpy as np

N = 256
MAX_SUM = 2 * N

def construct():
    start_time = time.time()
    deadline = start_time + 28.0  # Safety margin for 30s limit

    def is_sidon(A):
        """Check if A is a Sidon set."""
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(A):
        """Check if A is inclusion-maximal."""
        A_set = set(A)
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(1, N + 1):
            if x in A_set:
                continue
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for y in A:
                    if x + y in sums:
                        compatible = False
                        break
            if compatible:
                return False
        return True

    def greedy_np(init_A=None, randomize_ties=False, seed=None, bias=0.0):
        """
        Greedy saturation using NumPy for fast blocking count.
        """
        if seed is not None:
            random.seed(seed)
            
        A = list(init_A) if init_A else []
        A_set = set(A)
        
        # Initialize sum array
        sum_arr = np.zeros(MAX_SUM + 1, dtype=bool)
        for i in range(len(A)):
            for j in range(i, len(A)):
                sum_arr[A[i] + A[j]] = True
        
        unblocked = [x for x in range(1, N + 1) if x not in A_set]
        
        while unblocked:
            best_x = None
            best_score = -1
            best_new_sums = None
            best_new_sum_arr = None
            
            A_np = np.array(A) if A else np.array([])
            
            for x in unblocked:
                # Check compatibility with current A
                if sum_arr[2 * x]:
                    continue
                if len(A) > 0 and np.any(sum_arr[x + A_np]):
                    continue
                
                # Calculate new sums
                # new_sums = {2x} U {x+y | y in A}
                new_sums_list = [2 * x]
                if A:
                    new_sums_list.extend(x + y for y in A)
                new_sums_arr = np.unique(np.array(new_sums_list))
                
                # Create new sum_arr view (copy is fast enough for small arrays)
                new_sum_arr = sum_arr.copy()
                new_sum_arr[new_sums_arr] = True
                
                # Count newly blocked elements in unblocked
                # z is blocked if 2z in new_sums OR exists y in A U {x} s.t. z+y in new_sums
                z_candidates = np.array([z for z in unblocked if z != x])
                if len(z_candidates) == 0:
                    score = 0
                else:
                    # Diagonal check: 2z in new_sum_arr
                    blocked_diag = new_sum_arr[2 * z_candidates]
                    
                    # Off-diagonal check: z+y in new_sum_arr for some y in A U {x}
                    A_new = np.append(A_np, x) if len(A) > 0 else np.array([x])
                    # Create matrix of all sums z + y
                    # z_candidates shape (M,), A_new shape (K,)
                    # z_plus_y shape (M, K)
                    z_plus_y = z_candidates[:, None] + A_new[None, :]
                    # Check if any of these sums are in new_sum_arr
                    blocked_off = np.any(new_sum_arr[z_plus_y], axis=1)
                    
                    score = np.sum(blocked_diag | blocked_off)
                
                # Apply bias
                if bias != 0.0:
                    if bias > 0:
                        score -= x * bias
                    else:
                        score += x * abs(bias)
                
                # Apply randomization
                if randomize_ties:
                    score += random.uniform(0, 0.5)
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_sums = new_sums_arr
                    best_new_sum_arr = new_sum_arr

            if best_x is None:
                break
            
            # Update state
            A.append(best_x)
            A_set.add(best_x)
            sum_arr = best_new_sum_arr
            unblocked.remove(best_x)
            
        return A

    def local_search_improve(A):
        """
        Improve A by trying removals and swaps, followed by re-saturation.
        """
        A = list(A)
        improved = True
        while improved:
            improved = False
            current_size = len(A)
            
            # 1. Try removals
            for i in range(len(A)):
                if time.time() > deadline:
                    return A
                candidate = A[:i] + A[i+1:]
                if not candidate:
                    continue
                # Check Sidon (should be if A was Sidon)
                if not is_sidon(candidate):
                    continue
                saturated = greedy_np(init_A=candidate)
                if len(saturated) < current_size:
                    A = saturated
                    current_size = len(A)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try swaps (remove a, add b, then saturate)
            # Limit swaps to avoid excessive time
            for i in range(len(A)):
                if time.time() > deadline:
                    return A
                a = A[i]
                A_minus = A[:i] + A[i+1:]
                
                # Try a few random b's
                for _ in range(5):
                    b = random.randint(1, N)
                    if b in A_minus:
                        continue
                    # Check if A_minus U {b} is Sidon
                    candidate_swap = A_minus + [b]
                    if not is_sidon(candidate_swap):
                        continue
                    saturated = greedy_np(init_A=candidate_swap)
                    if len(saturated) < current_size:
                        A = saturated
                        current_size = len(A)
                        improved = True
                        break
                if improved:
                    break
            
        return A

    # Initial deterministic run
    best_set = greedy_np()
    best_size = len(best_set)
    
    # Apply local search
    improved = local_search_improve(best_set)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved

    # Randomized restarts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        r = random.random()
        
        # Diversify strategies based on random choice
        if r < 0.4:
            # Pure randomized greedy
            current_set = greedy_np(randomize_ties=True, seed=attempt)
        elif r < 0.6:
            # Start with random element
            init_x = random.randint(1, N)
            current_set = greedy_np(init_A=[init_x], randomize_ties=True, seed=attempt)
        elif r < 0.8:
            # Start with two random compatible elements
            x1 = random.randint(1, N)
            x2 = random.randint(1, N)
            while x2 == x1 or x1 + x2 == 2*x1 or x1 + x2 == 2*x2:
                x2 = random.randint(1, N)
            init = [x1, x2]
            if is_sidon(init):
                current_set = greedy_np(init_A=init, randomize_ties=True, seed=attempt)
            else:
                current_set = greedy_np(randomize_ties=True, seed=attempt)
        else:
            # Use bias to favor smaller or larger elements
            bias_val = random.uniform(0.01, 0.05) * (1 if random.random() < 0.5 else -1)
            current_set = greedy_np(randomize_ties=True, seed=attempt, bias=bias_val)
        
        # Apply local search if promising or periodically
        if len(current_set) <= best_size + 2:
            improved = local_search_improve(current_set)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    return best_set

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Size: {len(s)}")
    # Verify
    assert is_sidon(s), "Not Sidon"
    assert is_maximal(s), "Not Maximal"
