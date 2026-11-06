"""
Rewritten program: Local search for minimum color 4-AP coloring.
"""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 50  # Safety margin
    N = 128
    
    # Precompute all non-constant 4-term APs (a, b, c, d) with a < b < c < d and b-a = c-b = d-c
    ap_list = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < N:
                ap_list.append((a, b, c, e))
                
    num_aps = len(ap_list)
    
    def count_violations(colors, K):
        """Count number of 4-APs with < 3 distinct colors."""
        count = 0
        for a, b, c, e in ap_list:
            ca, cb, cc, ce = colors[a], colors[b], colors[c], colors[e]
            # Check if distinct colors < 3
            # This happens if:
            # 1. All 4 same (1 color)
            # 2. 3 same, 1 diff (2 colors)
            # 3. 2 same, 2 same, diff (2 colors)
            # 4. 2 same, 1 same, 1 same... wait.
            # Simply: len({ca, cb, cc, ce}) < 3
            
            # Optimization: 
            # If ca == cb == cc == ce: violation
            # If ca == cb == cc != ce: violation
            # If ca == cb != cc == ce: violation
            # If ca != cb == cc == ce: violation
            # If ca == cc == ce != cb: violation
            # If ca == cb != cc and ca != ce and cb != cc and cb != ce and cc != ce... no that's 4 colors.
            
            # Fast check:
            if ca == cb:
                if cc == ce:
                    # Pairs (ca, ca) and (cc, ce). If ca==cc, 1 color. If ca!=cc, 2 colors. Violation in both.
                    count += 1
                else:
                    # ca=cb, cc!=ce.
                    # Colors: {ca, cc, ce}. Size is 3 if ca not in {cc, ce} and cc!=ce.
                    # Size is 2 if ca==cc or ca==ce.
                    if ca == cc or ca == ce:
                        count += 1
            else:
                # ca != cb
                if cc == ce:
                    # ca!=cb, cc=ce.
                    # Colors {ca, cb, cc}. Size 3 if cc not in {ca, cb}.
                    # Size 2 if cc==ca or cc==cb.
                    if cc == ca or cc == cb:
                        count += 1
                else:
                    # ca!=cb, cc!=ce.
                    # Colors {ca, cb, cc, ce}.
                    # Size < 3 means size 1 or 2.
                    # Size 1 impossible since ca!=cb.
                    # Size 2 means exactly 2 distinct values.
                    # This implies {ca, cb} == {cc, ce} as sets.
                    # i.e. (ca==cc and cb==ce) or (ca==ce and cb==cc).
                    if (ca == cc and cb == ce) or (ca == ce and cb == cc):
                        count += 1
        return count

    def solve(K, max_iters=20000):
        """Try to find a valid K-coloring."""
        rng = random.Random(42)
        
        # Initialize randomly
        colors = [rng.randint(0, K - 1) for _ in range(N)]
        current_cost = count_violations(colors, K)
        
        if current_cost == 0:
            return colors
        
        best_cost = current_cost
        best_colors = colors[:]
        
        # Local Search
        T = 1.0  # Temperature
        cooling = 0.9995
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random position
            pos = rng.randint(0, N - 1)
            old_color = colors[pos]
            
            # Try a different color
            if K == 1:
                new_color = 0
            else:
                new_color = rng.randint(0, K - 1)
                if new_color == old_color:
                    new_color = (new_color + 1) % K
            
            colors[pos] = new_color
            new_cost = count_violations(colors, K)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    return colors
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-(delta)/T)
                delta = new_cost - current_cost
                if rng.random() < pow(2.718, -delta / T):
                    current_cost = new_cost
                else:
                    colors[pos] = old_color
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        if best_cost == 0:
            return best_colors
        return None

    # Try K=3 first
    res = solve(3, max_iters=50000)
    if res is not None:
        # Relabel to 0..2
        return res
    
    # Try K=4
    res = solve(4, max_iters=50000)
    if res is not None:
        # Relabel to 0..3
        # Ensure we use minimal labels
        mapping = {}
        new_res = []
        next_id = 0
        for c in res:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_res.append(mapping[c])
        return new_res
    
    # Fallback: Greedy 4-coloring (should always work)
    colors = [0] * N
    for i in range(N):
        forbidden = set()
        # Check constraints where i is involved
        for a, b, c, e in ap_list:
            if i in (a, b, c, e):
                others = [x for x in (a, b, c, e) if x != i]
                if all(colors[x] is not None for x in others):
                    oc = [colors[x] for x in others]
                    if len(set(oc)) < 3:
                        # If others have < 3 colors, i must add a new one or break the pattern?
                        # Actually, if others have 1 color {X}, i must be != X? No, if i=X, set is {X} (size 1). Violation.
                        # So if others have 1 color, i must be != that color?
                        # Wait. If others are {X, X, X}, and i is Y, set is {X, Y} (size 2). Violation.
                        # So if others have 1 color, NO color for i works?
                        # Let's re-read: "Every nonconstant four-term progression... must use AT LEAST three distinct colors."
                        # If a,b,c are all color 0, then {0,0,0,c_i} must have size >= 3.
                        # This is impossible. So a valid coloring cannot have 3 identical colors in a 4-AP prefix?
                        # Yes. So if we assign a,b,c to 0, we cannot assign i.
                        # This implies greedy might fail if it creates such a situation.
                        # But with 4 colors, we can avoid this.
                        # In greedy, we should forbid colors that lead to violation.
                        # If others have 1 color {X}, any i makes size 1 or 2. So no color is valid?
                        # This means the previous assignments were bad.
                        # To handle this in greedy, we just pick a color that minimizes future constraints, or restart.
                        # For now, if this happens, we just pick 0 and hope for the best, or use a smarter heuristic.
                        pass
        # Simple greedy: pick smallest color not causing immediate violation with *completed* APs?
        # This is tricky. Let's just use a known valid pattern for fallback.
        # 4-color periodic: 0,1,2,3, 0,1,2,3...
        # Check AP 0,1,2,3 -> 0,1,2,3 (4 colors) OK.
        # Check AP 0,2,4,6 -> 0,2,0,2 (2 colors) VIOLATION.
        # So periodic 4 is bad.
        
        # Use a random valid 4-coloring found by search if possible, otherwise hardcoded?
        # Let's try a different search if solve(4) failed.
        pass
        
    # If all else fails, return a random 4-coloring (likely invalid, but better than crash)
    # But we should ensure validity.
    # Let's try one more time with more iterations for K=4
    res = solve(4, max_iters=100000)
    if res is not None:
        mapping = {}
        new_res = []
        next_id = 0
        for c in res:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_res.append(mapping[c])
        return new_res

    # Ultimate fallback: 128 colors (trivially valid, each position unique)
    return list(range(N))

# EVOLVE-BLOCK-END
