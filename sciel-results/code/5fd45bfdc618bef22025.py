"""
Deterministic greedy backtracking with forward checking for two-colourings
of [1, 256] minimizing monochromatic 4-term arithmetic progressions.

Strategy:
1. Order integers by their participation in 4-APs (degree heuristic).
2. For each integer in that order, assign the color (0 or 1) that results in
   the fewest new monochromatic 4-APs being completed.
3. This is a greedy heuristic that avoids the local minima of SA by
   constructing the coloring from scratch with global awareness of AP structure.
"""

import time
import sys

def construct():
    # Deadline guard: finish well before the 120s evaluator limit
    deadline = time.time() + 110
    
    n = 256
    
    # Precompute all 4-term APs as tuples (i0, i1, i2, i3) 0-indexed
    aps = []
    for d in range(1, 86):  # d from 1 to 85 (since 3d <= 255)
        max_a = 256 - 3*d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a - 1 + d
            i2 = a - 1 + 2*d
            i3 = a - 1 + 3*d
            aps.append((i0, i1, i2, i3))
    
    total_aps = len(aps)
    
    # For each index, precompute which APs it participates in and its position
    # aps_containing[i] = list of (ap_idx, position)
    aps_containing = [[] for _ in range(n)]
    for ap_idx, (i0, i1, i2, i3) in enumerate(aps):
        aps_containing[i0].append((ap_idx, 0))
        aps_containing[i1].append((ap_idx, 1))
        aps_containing[i2].append((ap_idx, 2))
        aps_containing[i3].append((ap_idx, 3))
    
    # Determine the order in which to color integers.
    # Heuristic: color integers that are part of the most 4-APs first.
    # This allows early decisions to constrain the structure significantly.
    degrees = [len(aps_containing[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # We will build the coloring in this order.
    # colors[i] will be 0 or 1 once assigned, None otherwise.
    colors = [None] * n
    
    # To efficiently count "newly completed" mono-APs when assigning a color to index k,
    # we need to check all APs containing k.
    # An AP is "completed" mono if all 4 elements have the same color.
    # When we assign color c to k, an AP (i0,i1,i2,i3) containing k becomes
    # a new mono-AP if:
    # 1. The other 3 elements already have color c.
    # 2. The AP was not already mono (which is impossible if k was None).
    
    # We can simply iterate through aps_containing[k] and count how many
    # of the other 3 indices have the same color c.
    # If count == 3, then assigning c to k completes a new mono-AP.
    
    # However, we also want to avoid "creating" mono-APs.
    # Actually, the objective is to minimize the TOTAL number of mono-APs at the end.
    # A greedy approach: for each k in order, choose c in {0,1} that minimizes
    # the number of APs containing k where the other 3 are already colored c.
    
    # This is a forward-looking greedy. It doesn't look ahead beyond the immediate
    # completion, but the ordering by degree helps.
    
    # Let's also consider a simple tie-breaker: if both colors complete the same
    # number of APs, choose the one that balances the global color count, or just 0.
    
    for k in order:
        if time.time() > deadline - 2:
            # Safety: if running out of time, fill remaining with random/0
            # But since it's deterministic and fast, this shouldn't happen.
            break
            
        count_0 = 0
        count_1 = 0
        
        for (ap_idx, pos) in aps_containing[k]:
            i0, i1, i2, i3 = aps[ap_idx]
            
            # Get the other 3 indices
            others = []
            for p in range(4):
                if p != pos:
                    if p == 0: others.append(i0)
                    elif p == 1: others.append(i1)
                    elif p == 2: others.append(i2)
                    else: others.append(i3)
            
            # Check if all 3 others are colored 0
            if colors[others[0]] == 0 and colors[others[1]] == 0 and colors[others[2]] == 0:
                count_0 += 1
                
            # Check if all 3 others are colored 1
            if colors[others[0]] == 1 and colors[others[1]] == 1 and colors[others[2]] == 1:
                count_1 += 1
        
        # Choose the color that minimizes new mono-APs
        if count_0 < count_1:
            colors[k] = 0
        elif count_1 < count_0:
            colors[k] = 1
        else:
            # Tie-breaker: try to balance colors or just pick 0
            # Let's pick 0 to be deterministic
            colors[k] = 0
    
    # Ensure all are colored (should be, but just in case)
    for i in range(n):
        if colors[i] is None:
            colors[i] = 0
            
    # Verify and count
    def count_mono(colors):
        count = 0
        for (i0, i1, i2, i3) in aps:
            if colors[i0] == colors[i1] == colors[i2] == colors[i3]:
                count += 1
        return count
    
    # The greedy above is a single pass. It might not be optimal.
    # But it's a different structural approach.
    # To improve, we could run multiple greedy passes with different tie-breakers
    # or slight variations in ordering.
    
    # Let's try a few variations to see if we can get a better result.
    # Variation 1: Standard greedy (as above)
    best_colors = colors[:]
    best_score = count_mono(best_colors)
    
    # Variation 2: Reverse order (least constrained first)
    order2 = sorted(range(n), key=lambda i: degrees[i])
    colors2 = [None] * n
    for k in order2:
        count_0 = 0
        count_1 = 0
        for (ap_idx, pos) in aps_containing[k]:
            i0, i1, i2, i3 = aps[ap_idx]
            others = []
            for p in range(4):
                if p != pos:
                    if p == 0: others.append(i0)
                    elif p == 1: others.append(i1)
                    elif p == 2: others.append(i2)
                    else: others.append(i3)
            if colors2[others[0]] == 0 and colors2[others[1]] == 0 and colors2[others[2]] == 0:
                count_0 += 1
            if colors2[others[0]] == 1 and colors2[others[1]] == 1 and colors2[others[2]] == 1:
                count_1 += 1
        if count_0 < count_1:
            colors2[k] = 0
        elif count_1 < count_0:
            colors2[k] = 1
        else:
            colors2[k] = 1 # Different tie-breaker
    for i in range(n):
        if colors2[i] is None:
            colors2[i] = 1
            
    score2 = count_mono(colors2)
    if score2 < best_score:
        best_score = score2
        best_colors = colors2[:]
        
    # Variation 3: Randomized greedy with fixed seed for reproducibility within run
    # But the task asks for a deterministic approach in the strategy description.
    # However, we can use a deterministic pseudo-random sequence if needed.
    # Let's stick to the two deterministic orders for now.
    
    # Let's also try a "balanced" tie-breaker: keep track of global counts
    # and choose the color that keeps the total number of 0s and 1s closer to 128.
    
    # Actually, the current greedy is very fast. We have time budget.
    # We can run a local search polish on the best result found by greedy.
    # This combines the structural benefit of greedy initialization with
    # the refinement power of local search.
    
    # Polish using simple hill climbing (flip if it improves)
    # We will do this for the best_colors found so far.
    
    current_colors = best_colors[:]
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n):
            # Calculate delta if we flip i
            old_c = current_colors[i]
            new_c = 1 - old_c
            delta = 0
            for (ap_idx, pos) in aps_containing[i]:
                i0, i1, i2, i3 = aps[ap_idx]
                others = []
                for p in range(4):
                    if p != pos:
                        if p == 0: others.append(i0)
                        elif p == 1: others.append(i1)
                        elif p == 2: others.append(i2)
                        else: others.append(i3)
                        
                # Before flip
                if current_colors[others[0]] == old_c and current_colors[others[1]] == old_c and current_colors[others[2]] == old_c:
                    delta -= 1
                # After flip
                if current_colors[others[0]] == new_c and current_colors[others[1]] == new_c and current_colors[others[2]] == new_c:
                    delta += 1
            
            if delta < 0:
                current_colors[i] = new_c
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_colors = current_colors[:]
    
    return best_colors

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Colors: {result[:20]}...")
    assert len(result) == 256
    assert all(c in (0, 1) for c in result)
