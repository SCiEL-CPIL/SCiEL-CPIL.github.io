import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    MAX_VAL = 1000000
    
    start_time = time.time()
    deadline = 85.0  # Safety margin

    # Precompute residues for each divisor
    # residues[i] is a list of r in [0, LCM) such that r % divisors[i] == 0
    residues = []
    for d in divisors:
        r_list = []
        for r in range(0, LCM, d):
            r_list.append(r)
        residues.append(r_list)

    # To speed up window checking, we can sort residues and use binary search
    # But since we iterate W, and L, we can just check membership in a set or
    # use the fact that the window is small.
    
    # For each divisor, create a boolean array or set for fast lookup?
    # Actually, for a given L and W, the candidates for divisor i are
    # r in residues[i] such that (r - L) % LCM <= W.
    
    # Let's iterate W from 1 upwards
    for W in range(1, 200):
        if time.time() - start_time > deadline:
            # Fallback: return divisors
            return [int(d) for d in divisors]
        
        # Iterate over all starting positions L in the circular domain
        # Due to symmetry, we might not need to check all L, but LCM is only 30030,
        # and the inner check is fast if it fails early.
        for L in range(LCM):
            if time.time() - start_time > deadline:
                return [int(d) for d in divisors]
            
            # Determine candidates for each divisor in the window [L, L+W] (circular)
            candidates = []
            feasible = True
            for i in range(n):
                d = divisors[i]
                opts = []
                # The window is [L, L+W] modulo LCM.
                # A residue r is in the window if (r - L) % LCM <= W.
                # We can iterate through the residues of d and check this condition.
                # Optimization: The first residue >= L is r0 = (L // d + 1) * d if L % d != 0 else L.
                # Actually, just finding the first multiple >= L and stepping by d.
                
                # First multiple >= L
                # k = ceil(L / d)
                # if L % d == 0: k = L // d else: k = L // d + 1
                k = (L + d - 1) // d
                r = k * d
                if r >= LCM:
                    r -= LCM # Wrap around? No, r is in [0, LCM).
                    # Wait, k is based on L. If L is large, r might be >= LCM.
                    # But we are working modulo LCM.
                    # Let's just check the condition (r - L) % LCM <= W for the generated multiples.
                    
                # Generate multiples in the window
                # The window size is W+1.
                # Number of multiples is approx (W+1)/d.
                
                # Efficient check:
                # The valid residues are r such that L <= r <= L+W (mod LCM).
                # This is equivalent to r in [L, L+W] or (if wrapping) r in [0, L+W-LCM] U [L, LCM-1].
                
                # Simpler: just check the few candidates around L.
                # The candidates for d are spaced by d.
                # Start from the first multiple >= L.
                # If L % d != 0, first multiple is L - (L % d) + d.
                # Let's just use the property: r % d == 0.
                
                # We can just iterate the residues list and filter?
                # No, residues list is long.
                
                # Let's calculate the specific residues in the window.
                # The window covers values [L, L+W].
                # Any multiple of d in this range is a candidate.
                # Let first_mult = ((L + d - 1) // d) * d
                # If first_mult > L + W (and considering wrap), no candidates.
                
                # Handle wrap-around carefully.
                # Case 1: No wrap. L+W < LCM.
                #   Candidates: r in [L, L+W] s.t. r % d == 0.
                # Case 2: Wrap. L+W >= LCM.
                #   Candidates: r in [L, LCM-1] s.t. r % d == 0, AND r in [0, L+W-LCM] s.t. r % d == 0.
                
                if L + W < LCM:
                    # No wrap
                    start_r = ((L + d - 1) // d) * d
                    if start_r > L + W:
                        feasible = False
                        break
                    r = start_r
                    while r <= L + W:
                        opts.append(r)
                        r += d
                else:
                    # Wrap
                    # Part 1: [L, LCM-1]
                    start_r1 = ((L + d - 1) // d) * d
                    if start_r1 < LCM:
                        r = start_r1
                        while r < LCM:
                            opts.append(r)
                            r += d
                    # Part 2: [0, L+W-LCM]
                    end_r2 = L + W - LCM
                    if end_r2 >= 0:
                        # Last multiple <= end_r2
                        # k = floor(end_r2 / d)
                        k = end_r2 // d
                        r = k * d
                        while r >= 0:
                            opts.append(r)
                            r -= d
                        # Reverse to keep sorted? Not strictly necessary for matching, but good for determinism
                        opts.reverse()
                    
                    if not opts:
                        feasible = False
                        break

                if not opts:
                    feasible = False
                    break
                candidates.append(opts)
            
            if not feasible:
                continue
            
            # We have candidates for each divisor. Now solve the matching problem.
            # Assign distinct values from candidates[i] to divisor i.
            # Use backtracking with forward checking.
            # Sort divisors by number of candidates (most constrained first)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                i = order[idx]
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        
                        # Forward checking
                        feasible = True
                        for j in range(idx + 1, n):
                            k = order[j]
                            has_option = False
                            for v in candidates[k]:
                                if v not in used:
                                    has_option = True
                                    break
                            if not has_option:
                                feasible = False
                                break
                        
                        if feasible and backtrack(idx + 1):
                            return True
                        
                        used.remove(val)
                        assignment[i] = None
                
                return False
            
            if backtrack(0):
                # We found a valid set of residues
                # Now we need to lift them to x_i <= 1,000,000
                # The residues are in [0, LCM).
                # We can add k * LCM to each residue.
                # To minimize the range, we already fixed the circular span W.
                # The linear range will be W, unless the wrap-around causes a jump in the linear domain.
                # However, the problem asks for min(max) - min(min).
                # If we pick the residues, the linear values are just the residues themselves (if we pick k=0)
                # OR we can shift them.
                # But wait, if the window wrapped around 0, the linear span of the residues might be large (e.g., 30000).
                # Example: L=29990, W=10. Residues: 29995, 29999, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
                # Linear span: 29999 - 0 = 29999.
                # But circular span is 10.
                # The problem asks for min(max(x) - min(x)).
                # If we use the residues directly, the range is large.
                # But we can shift the whole set by k*LCM.
                # If we shift by k*LCM, the range remains the same.
                # So if the circular span is W, can we always achieve a linear span of W?
                # Yes, if we "cut" the circle at a point not occupied by any x_i, the linear span is W.
                # However, the x_i must be positive integers.
                # If we use the residues, they are in [0, LCM-1].
                # If the window wraps, the residues are "split" into high and low values.
                # To get a linear span of W, we need to map the "high" values to the next cycle or "low" to previous?
                # No, we just pick specific integers.
                # If we have residues $r_1, \dots, r_n$ in a circular window of size W.
                # We can choose integers $x_i$ such that $x_i \equiv r_i \pmod{LCM}$.
                # We want $\max x_i - \min x_i$ to be minimal.
                # This is equivalent to finding a shift $S$ such that the values $r_i + k_i LCM$ are as close as possible.
                # Actually, if the circular span is W, we can always find a set of integers with linear span W.
                # Proof: Let the residues be sorted $r_{(1)} < \dots < r_{(n)}$ on the circle.
                # If they don't wrap, $x_i = r_i$ works, span is $W$.
                # If they wrap, say $r_{(1)} \dots r_{(k)}$ are near LCM and $r_{(k+1)} \dots r_{(n)}$ are near 0.
                # We can set $x_i = r_i$ for the low ones, and $x_i = r_i - LCM$ for the high ones? No, must be positive.
                # We can set $x_i = r_i + LCM$ for the low ones?
                # Let's just pick the specific integers that form the tightest cluster.
                # Given the residues, we can try all $n$ possible "cuts" (shifting one subset by LCM) to find the min linear span.
                # Or simpler: The linear span of the optimal solution for a given circular window is exactly W.
                # We just need to output valid integers.
                
                # Let's construct the integers.
                # We have assignment[i] = residue r_i.
                # We want to choose $x_i = r_i + k_i LCM$ to minimize $\max x_i - \min x_i$.
                # Since the circular span is W, the optimal linear span is W.
                # We can achieve this by picking a reference point.
                # Let's just take the residues and check the span.
                # If the span is > W, it means it wrapped.
                # If it wrapped, we can add LCM to the small values (or subtract from large, but must stay positive).
                # Since we can add any multiple of LCM, we can shift the "small" values up by LCM to be next to the "large" ones.
                
                # Let's find the min and max of the residues.
                # If max - min <= W, we are good.
                # If max - min > W, we have a wrap.
                # In that case, we can add LCM to the residues that are "small" (less than some threshold) to bring them up.
                
                # Actually, a robust way:
                # Try all possible shifts for each element? No, that's $N^{10}$.
                # But we only need to shift the "cluster".
                # The residues lie in an interval of length W on the circle.
                # We can map them to an interval of length W on the line.
                
                # Let's identify the "break" point.
                # Sort the residues.
                # If the gap between consecutive sorted residues (circular) is > W, then the break is there.
                # The residues are in [L, L+W] mod LCM.
                
                # Simpler approach:
                # We found a valid assignment of residues.
                # Let's just return the residues as integers, BUT if the range is large, adjust.
                # However, the evaluator just checks $x_i \le 1000000$ and distinct.
                # And minimizes the range.
                # So we MUST return the tightest linear span.
                
                # Let's compute the optimal linear span for this set of residues.
                # We can try all $n$ choices for which residue is the "minimum".
                # If we fix $x_{min}$ to be based on residue $r_j$, we can set all other $x_i$ to be the closest multiple to $r_j + \text{offset}$.
                # Actually, just:
                # For each $j \in 0..n-1$, assume $x_j$ is the minimum.
                # Then $x_i$ should be the smallest multiple of $LCM$ added to $r_i$ such that $x_i \ge x_j$.
                # This is getting complicated.
                
                # Alternative:
                # The circular span is W.
                # The linear span is W if we cut the circle at a point not in the set.
                # We can just output the integers that form the contiguous block of size W.
                
                # Let's extract the integers.
                # We have residues $r_i$.
                # We want to assign an integer $k_i$ to each such that $x_i = r_i + k_i LCM$.
                # We want to minimize $\max x_i - \min x_i$.
                
                # Since the residues are within a circular window of size W,
                # we can just pick the representative in the window $[S, S+W]$ for some S.
                # We know such a window exists.
                # We can find S by looking at the residues.
                # Let's just try all $n$ possible "anchors".
                # Anchor at $r_j$. Let $S = r_j$.
                # For each $i$, find the multiple $x_i$ of $LCM$ added to $r_i$ such that $x_i \in [r_j, r_j + W]$.
                # If such $x_i$ exists for all $i$, and they are distinct, we have a solution with span $W$.
                
                found_solution = False
                for j in range(n):
                    anchor = assignment[j]
                    # Window is [anchor, anchor + W]
                    # Check if all other residues can be mapped to this window
                    xs = []
                    valid = True
                    for i in range(n):
                        r = assignment[i]
                        # We need x = r + k*LCM such that anchor <= x <= anchor + W
                        # k*LCM in [anchor - r, anchor + W - r]
                        low_k = (anchor - r + LCM - 1) // LCM
                        high_k = (anchor + W - r) // LCM
                        # Check if there is an integer k in [low_k, high_k]
                        if low_k <= high_k:
                            # Pick k = low_k (smallest x)
                            x = r + low_k * LCM
                            xs.append(x)
                        else:
                            valid = False
                            break
                    if valid:
                        # Check distinctness
                        if len(set(xs)) == n:
                            # Check bounds
                            if all(0 < x <= MAX_VAL for x in xs):
                                # We have a valid solution with span <= W
                                # Return it
                                return [int(x) for x in xs]
                            # If out of bounds, we might be able to shift the whole set by -LCM?
                            # But we want positive.
                            # If max > MAX_VAL, we can't shift down if min becomes <= 0.
                            # But we can try other anchors or other W.
    
    # Fallback
    return [int(d) for d in divisors]

# Test locally? No, just output.
