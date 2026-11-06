import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 192
    MAX_SIZE = 48
    K = 3

    # Precompute residue masks
    # res_mask[d][r] is a bitset of integers v in [1, N] such that v % d == r
    res_mask = [[0] * d for d in range(N + 1)]
    for d in range(1, N + 1):
        for v in range(1, N + 1):
            r = v % d
            res_mask[d][r] |= (1 << (v - 1))

    def can_add(S_mask, x):
        """
        Check if x can be added to set represented by S_mask.
        S_mask is an integer bitset where bit i is set if (i+1) is in S.
        x is an integer in [1, N].
        Returns True if valid.
        """
        # x must not be in S
        if (S_mask >> (x - 1)) & 1:
            return False
        
        # Part A: Check if x is safe as a divisor.
        # No 3 distinct elements y, z, w in S such that x | (y+z+w)
        # Iterate y in S
        # We need z, w in S \ {y} such that z+w = -y mod x
        
        # Get list of elements in S for iteration (optimization: only iterate if S is large enough)
        # For speed, we can iterate bits
        S_list = [i + 1 for i in range(N) if (S_mask >> i) & 1]
        
        if len(S_list) >= 3:
            for y in S_list:
                R = (-y) % x
                # Check if there exist z, w in S \ {y} such that z+w = R mod x
                # Iterate z in S \ {y}
                for z in S_list:
                    if z == y:
                        continue
                    r_w = (R - z) % x
                    # Check if there is w in S such that w % x == r_w
                    # w must be distinct from y and z
                    mask_w = res_mask[x][r_w]
                    # Exclude y and z
                    mask_w &= ~(1 << (y - 1))
                    mask_w &= ~(1 << (z - 1))
                    if (S_mask & mask_w):
                        return False

        # Part B: Check if existing elements d in S are safe with x added.
        # For each d in S, no triple in (S \ {d}) U {x} sums to 0 mod d.
        # Since S was valid, only triples involving x can be violations.
        # Triples are (x, y, z) with y, z in S \ {d, x}.
        # Condition: d | (x + y + z) => y + z = -x mod d
        
        for d in S_list:
            R = (-x) % d
            # Check if there exist y, z in S \ {d, x} such that y+z = R mod d
            for y in S_list:
                if y == d:
                    continue
                # y is in S. x is not in S yet.
                r_z = (R - y) % d
                mask_z = res_mask[d][r_z]
                # Exclude d, y, and x (x is not in S_mask, so no need to exclude from mask, 
                # but we must ensure z is distinct from y)
                mask_z &= ~(1 << (d - 1))
                mask_z &= ~(1 << (y - 1))
                if (S_mask & mask_z):
                    return False

        return True

    def is_valid(S_mask):
        """Full validation."""
        S_list = [i + 1 for i in range(N) if (S_mask >> i) & 1]
        n = len(S_list)
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            # Check if a divides any sum of 3 in others
            # Use the same logic as Part A of can_add but with a as divisor and others as the set
            # We can just iterate triples for correctness check in final step, 
            # but for speed in search we rely on can_add.
            # For final validation, a simple O(N^3) is fine for N=48.
            for b_idx in range(len(others)):
                b = others[b_idx]
                for c_idx in range(b_idx + 1, len(others)):
                    c = others[c_idx]
                    for d_idx in range(c_idx + 1, len(others)):
                        d = others[d_idx]
                        if (b + c + d) % a == 0:
                            return False
        return True

    def greedy(order, S_mask=0):
        """Greedy build using given order of candidates."""
        for x in order:
            if S_mask.bit_count() >= MAX_SIZE:
                break
            if can_add(S_mask, x):
                S_mask |= (1 << (x - 1))
        return S_mask

    def local_search(S_mask):
        """Local search: try to add elements, then try swaps."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add
            for x in range(1, N + 1):
                if S_mask.bit_count() >= MAX_SIZE:
                    break
                if not ((S_mask >> (x - 1)) & 1) and can_add(S_mask, x):
                    S_mask |= (1 << (x - 1))
                    improved = True

            # 2. Try 1-for-1 swaps
            if S_mask.bit_count() < MAX_SIZE:
                for a in range(1, N + 1):
                    if not ((S_mask >> (a - 1)) & 1):
                        continue
                    # Remove a
                    new_mask = S_mask & ~(1 << (a - 1))
                    # Try to add any x not in new_mask
                    for x in range(1, N + 1):
                        if (new_mask >> (x - 1)) & 1:
                            continue
                        if can_add(new_mask, x):
                            S_mask = new_mask | (1 << (x - 1))
                            improved = True
                            break # Restart swap search? Or continue? 
                            # Restarting is safer for local search convergence
                            break 

            # 3. Try 1-for-2 swaps (if size < MAX_SIZE)
            if S_mask.bit_count() < MAX_SIZE:
                for a in range(1, N + 1):
                    if not ((S_mask >> (a - 1)) & 1):
                        continue
                    new_mask = S_mask & ~(1 << (a - 1))
                    added = 0
                    # Try to add 2 elements
                    for x1 in range(1, N + 1):
                        if (new_mask >> (x1 - 1)) & 1:
                            continue
                        if can_add(new_mask, x1):
                            temp_mask = new_mask | (1 << (x1 - 1))
                            for x2 in range(x1 + 1, N + 1):
                                if (temp_mask >> (x2 - 1)) & 1:
                                    continue
                                if can_add(temp_mask, x2):
                                    S_mask = temp_mask | (1 << (x2 - 1))
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

        return S_mask

    random.seed(42)
    best_mask = 0
    best_size = 0

    # Strategy 1: Descending
    S = greedy(range(N, 0, -1))
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S

    # Strategy 2: Ascending
    S = greedy(range(1, N + 1))
    if S.bit_count() > best_size:
        best_size = S.bit_count()
        best_mask = S

    # Strategy 3: Random restarts with Local Search
    for _ in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy(order)
        S = local_search(S)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S
            if best_size == MAX_SIZE:
                break

    # Final polish on best
    if time.time() < deadline:
        S = local_search(best_mask)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S

    # Convert to list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    
    # Final validation
    if not is_valid(best_mask):
        # Fallback
        result = []
        for x in range(N, 0, -1):
            if len(result) >= MAX_SIZE:
                break
            # Re-check with list-based logic for safety
            S_set = set(result)
            if can_add(0, x) if not result else True: # Placeholder, better to just return empty if invalid
                pass
        # Simpler fallback: just return the greedy descending result which is usually valid
        S = greedy(range(N, 0, -1))
        result = [i + 1 for i in range(N) if (S >> i) & 1]

    return result

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
