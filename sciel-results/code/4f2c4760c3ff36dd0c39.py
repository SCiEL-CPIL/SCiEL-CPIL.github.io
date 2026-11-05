import time
import random

def construct():
    t_start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    n = 256

    def is_valid(S_list):
        S_list = sorted(S_list)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S_sorted, x):
        """Check if adding x to S_sorted maintains the property.
        x must be smaller than all elements in S (we add smaller elements).
        Need: for all b, c in S with b <= c, x does NOT divide b+c."""
        if x <= 0:
            return False
        # Build residue counts mod x
        residues = {}
        for v in S_sorted:
            r = v % x
            residues[r] = residues.get(r, 0) + 1
        
        for r, cnt in residues.items():
            complement = (x - r) % x
            if r == 0:
                # Need two elements with residue 0 (b=c case where b ≡ 0 mod x)
                if cnt >= 2:
                    return False
            else:
                # Need one with residue r and one with residue complement
                if complement in residues and residues[complement] >= 1:
                    return False
        return True

    def greedy_fill(S_sorted):
        """Try to add any missing element from 1..n to S_sorted."""
        S_set = set(S_sorted)
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in S_set:
                    if can_add(S_sorted, x):
                        S_sorted.append(x)
                        S_set.add(x)
                        S_sorted.sort()
                        changed = True
                        break
        return S_sorted

    # Base: [171, 256], size 86
    base_S = list(range(171, n + 1))
    best_solution = list(base_S)
    best_size = len(base_S)

    # Strategy 1: Greedy from base, adding smaller elements
    S1 = list(base_S)
    S1_sorted = sorted(S1)
    for x in range(170, 0, -1):
        if time.time() - t_start > DEADLINE:
            break
        if can_add(S1_sorted, x):
            S1.append(x)
            S1_sorted.sort()
    S1 = greedy_fill(S1_sorted)
    if len(S1) > best_size:
        best_size = len(S1)
        best_solution = list(S1)

    # Strategy 2: Try different base intervals [k, 256]
    for k in range(150, 175):
        if time.time() - t_start > DEADLINE:
            break
        S2 = list(range(k, n + 1))
        S2_sorted = sorted(S2)
        for x in range(k - 1, 0, -1):
            if time.time() - t_start > DEADLINE:
                break
            if can_add(S2_sorted, x):
                S2.append(x)
                S2_sorted.sort()
        S2 = greedy_fill(S2_sorted)
        if len(S2) > best_size:
            best_size = len(S2)
            best_solution = list(S2)

    # Strategy 3: Local search with random restarts and swaps
    random.seed(42)
    for restart in range(60):
        if time.time() - t_start > DEADLINE:
            break

        # Start from best or perturbation
        S_cur = list(best_solution)
        if restart > 0 and len(S_cur) > 20:
            # Remove 1-3 random elements
            for _ in range(random.randint(1, 3)):
                if S_cur:
                    idx = random.randint(0, len(S_cur) - 1)
                    S_cur.pop(idx)

        S_cur_sorted = sorted(S_cur)
        S_cur_set = set(S_cur)

        # Phase 1: Greedy fill
        S_cur = greedy_fill(S_cur_sorted)
        S_cur_sorted = sorted(S_cur)
        S_cur_set = set(S_cur)

        # Phase 2: Iterative swap improvements
        for iteration in range(40):
            if time.time() - t_start > DEADLINE:
                break
            improved = False
            
            # Try removing one element and adding another
            for y in list(S_cur):
                if time.time() - t_start > DEADLINE:
                    break
                S_temp = [v for v in S_cur if v != y]
                S_temp_set = set(S_temp)
                S_temp_sorted = sorted(S_temp)
                
                # Try to add any element not in S_temp
                for x in range(1, n + 1):
                    if x not in S_temp_set and x != y:
                        if can_add(S_temp_sorted, x):
                            S_cur = S_temp + [x]
                            S_cur_set = set(S_cur)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break

        if len(S_cur) > best_size:
            best_size = len(S_cur)
            best_solution = list(S_cur)

    # Strategy 4: Final greedy fill on best solution
    S_final = list(best_solution)
    S_final_sorted = sorted(S_final)
    S_final = greedy_fill(S_final_sorted)

    if len(S_final) > best_size:
        best_size = len(S_final)
        best_solution = list(S_final)

    # Final verification
    S_final = sorted(best_solution)
    if not is_valid(S_final):
        S_final = list(base_S)

    return S_final
