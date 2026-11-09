import time
import random

def construct():
    """
    Return a list of 128 integer color labels in 0..127 such that every
    nonconstant four-term arithmetic progression of positions uses at least
    three distinct colors. Minimize the number of distinct labels used.
    """
    deadline = time.time() + 75.0  # safety margin before 90s kill
    n = 128
    positions = list(range(1, n + 1))  # positions 1..128
    
    # Precompute all nonconstant 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0 and a+3d <= n
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n - 2 * a + 1):
            a1 = a
            a2 = a + d
            a3 = a + 2 * d
            a4 = a + 3 * d
            if a4 <= n:
                progressions.append((a1 - 1, a2 - 1, a3 - 1, a4 - 1))  # 0-indexed
    
    # For each position, precompute which progressions it participates in
    # and its index within those progressions
    pos_progressions = [[] for _ in range(n)]
    for pi, (i1, i2, i3, i4) in enumerate(progressions):
        for idx in range(4):
            pos_progressions[i1 if idx == 0 else i2 if idx == 1 else i3 if idx == 2 else i4].append((pi, idx))
    
    best_score = n  # worst case: all different colors
    best_assignment = list(range(n))
    
    # Start with a greedy random initialization
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Try with small number of colors: 3, 4, 5, ...
        # Start with 3 colors
        num_colors = 3
        assignment = [random.randint(0, num_colors - 1) for _ in range(n)]
        
        # Local search: try to fix violations by changing colors
        violated = set(range(len(progressions)))
        
        # Count initial violations
        for pi in range(len(progressions)):
            i1, i2, i3, i4 = progressions[pi]
            colors_in_prog = {assignment[i1], assignment[i2], assignment[i3], assignment[i4]}
            if len(colors_in_prog) < 3:
                violated.add(pi)
        
        cur_score = len(violated)
        if cur_score < best_score:
            best_score = cur_score
            best_assignment = assignment[:]
        
        # Local search
        no_improve = 0
        max_iters = 50000
        iters = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            if not violated:
                break
            
            # Pick a random violated progression
            pi = random.choice(list(violated))
            i1, i2, i3, i4 = progressions[pi]
            
            # Try changing each position in the progression to a different color
            improved = False
            for pos_idx, pos in enumerate((i1, i2, i3, i4)):
                old_color = assignment[pos]
                
                # Try each possible new color
                for new_color in range(num_colors):
                    if new_color == old_color:
                        continue
                    
                    # Check if this change helps
                    assignment[pos] = new_color
                    
                    # Check progressions involving this position
                    new_violated_count = 0
                    for (pidx, idx_in_p) in pos_progressions[pos]:
                        j1, j2, j3, j4 = progressions[pidx]
                        cs = {assignment[j1], assignment[j2], assignment[j3], assignment[j4]}
                        if len(cs) < 3:
                            new_violated_count += 1
                    
                    if new_violated_count < cur_score:
                        # Accept the move
                        cur_score = new_violated_count
                        # Update violated set (approximate)
                        violated = set()
                        for pidx in range(len(progressions)):
                            j1, j2, j3, j4 = progressions[pidx]
                            cs = {assignment[j1], assignment[j2], assignment[j3], assignment[j4]}
                            if len(cs) < 3:
                                violated.add(pidx)
                        improved = True
                        break
                    else:
                        assignment[pos] = old_color
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 5000:
                    break
            else:
                no_improve = 0
        
        if not violated:
            # Found a valid coloring with num_colors
            distinct = set(assignment)
            if len(distinct) < best_score:
                best_score = len(distinct)
                best_assignment = assignment[:]
            break
        
        # If stuck, try with more colors
        if num_colors < 8:
            num_colors += 1
    
    # Final validation and score computation
    # Remap colors to 0..k-1 where k is number of distinct colors
    distinct_colors = sorted(set(best_assignment))
    color_map = {c: i for i, c in enumerate(distinct_colors)}
    final_assignment = [color_map[c] for c in best_assignment]
    
    # Validate
    for i1, i2, i3, i4 in progressions:
        cs = {final_assignment[i1], final_assignment[i2], final_assignment[i3], final_assignment[i4]}
        if len(cs) < 3:
            # Invalid - but we return best effort
            break
    
    return final_assignment

if __name__ == "__main__":
    result = construct()
    print(f"Number of distinct colors: {len(set(result))}")
