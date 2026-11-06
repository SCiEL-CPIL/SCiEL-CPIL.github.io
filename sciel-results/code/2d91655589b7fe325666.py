import time
import random
import itertools
import sys

_DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def construct():
    n = 16
    # Edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    E = len(edges) # 120
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
        
    K4_COUNT = len(k4s) # 1820
    
    # Map edge to list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)

    # Initialize colors: (i + j) % 17
    colors = [0] * E
    for (i, j), k in edge_idx.items():
        colors[k] = (i + j) % 17
        
    # State for incremental evaluation
    # k4_counts[k] is a dict or list of counts for each color in K4 k
    # Since colors are small, let's use a dict for sparsity or a fixed-size array if we bound colors.
    # Let's use a dict for safety, or just recompute distinct count for affected K4s.
    # Recomputing distinct count for ~20 K4s is very fast (20 * 6 edges = 120 ops).
    # Let's stick to the simpler "recompute affected" approach first to ensure correctness,
    # but optimize the inner loop.
    
    def get_k4_distinct(k4_idx):
        e_list = k4s[k4_idx]
        c_set = set()
        for e in e_list:
            c_set.add(colors[e])
        return len(c_set)

    def total_badness():
        total = 0
        for k in range(K4_COUNT):
            d = get_k4_distinct(k)
            if d < 5:
                total += (5 - d)
        return total

    def palette_size():
        return len(set(colors))

    current_badness = total_badness()
    current_palette = palette_size()
    
    best_colors = colors[:]
    best_badness = current_badness
    best_palette = current_palette
    
    # If initial is invalid (shouldn't be), we need to fix it.
    # (i+j)%17 for K4 {a,b,c,d}:
    # Edges: a+b, a+c, a+d, b+c, b+d, c+d
    # If two edges have same color, say a+b = a+c => b=c (impossible)
    # a+b = b+c => a=c (impossible)
    # a+b = c+d => a+b = c+d mod 17.
    # It is possible to have only 4 distinct colors?
    # The seed program assumes it's valid. Let's trust the math:
    # For (i+j) mod p, a K4 has 4 vertices. The 6 sums are distinct mod p if p is large enough?
    # Actually, for p=17, it's known to be a valid coloring for this constraint?
    # The prompt says "Every set of four vertices must span edges of at least five distinct colours."
    # With 17 colors, it's very likely valid.
    
    rng = random.Random(42)
    
    max_iter = 5000000
    iter_count = 0
    
    # We will use a simple hill climbing / simulated annealing hybrid.
    # Objective: Minimize (badness * 10000 + palette_size)
    
    # To speed up, we will maintain a list of "bad" K4s if badness > 0.
    # But for now, let's just do random edge swaps with full check on affected K4s.
    
    while _remaining() > 5.0 and iter_count < max_iter:
        iter_count += 1
        
        # Choose a move
        if current_badness > 0:
            # Try to fix a violation
            # Pick a random bad K4, then one of its edges
            # For simplicity, pick a random edge and random color
            e = rng.randrange(E)
            old_c = colors[e]
            # Try a few random colors
            candidates = [rng.randint(0, 30) for _ in range(5)]
            candidates.append(old_c) # to allow no-op? No, we want change
            candidates = [c for c in candidates if c != old_c]
            if not candidates:
                continue
            rng.shuffle(candidates)
        else:
            # Try to reduce palette
            # Pick a random edge
            e = rng.randrange(E)
            old_c = colors[e]
            # Try to change to an existing color to potentially eliminate old_c
            pal = list(set(colors))
            candidates = [c for c in pal if c != old_c]
            if not candidates:
                continue
            # Bias towards colors that are more frequent? Or just random.
            rng.shuffle(candidates)
            # Also, occasionally try a new color to escape local minima?
            # If we are stuck at palette size P, maybe we need to increase it temporarily?
            # Unlikely to help minimize, but let's stick to existing colors first.
            # If we want to eliminate old_c, we should check if count(old_c) becomes 0.
            # We'll handle acceptance logic below.
            
        best_move_c = old_c
        best_move_score = current_badness * 10000 + current_palette
        best_move_badness = current_badness
        best_move_palette = current_palette
        
        # Limit candidates to speed up
        for new_c in candidates[:10]: # Check only top 10 candidates
            if new_c == old_c:
                continue
                
            # Apply move
            colors[e] = new_c
            
            # Evaluate affected K4s
            new_badness = 0
            # We need to know the old badness contribution of affected K4s to update efficiently?
            # Or just recompute badness for affected K4s and adjust global?
            # Since we don't store per-K4 badness, let's just recompute total badness?
            # That's O(K4_COUNT * 6). 1820 * 6 = 10k ops.
            # 5000000 iters * 10k ops = 50 billion ops. Too slow.
            
            # We MUST use incremental updates.
            # Let's refactor to store k4_distinct and update it.
            break # Break out of candidate loop to implement proper incremental logic in a helper
            
    # The above loop was a sketch. Let's implement the proper incremental version.
    
    # Reset colors to initial
    for (i, j), k in edge_idx.items():
        colors[k] = (i + j) % 17
        
    # Precompute initial state
    k4_distinct = [0] * K4_COUNT
    for k in range(K4_COUNT):
        k4_distinct[k] = get_k4_distinct(k)
        
    # Store counts of colors per K4? 
    # If we just store distinct count, we need to know if a color disappears or appears.
    # So we need counts.
    # k4_counts[k] = {color: count}
    k4_counts = [{} for _ in range(K4_COUNT)]
    for k in range(K4_COUNT):
        for e in k4s[k]:
            c = colors[e]
            k4_counts[k][c] = k4_counts[k].get(c, 0) + 1
        k4_distinct[k] = len(k4_counts[k])
        
    def calc_badness():
        b = 0
        for k in range(K4_COUNT):
            if k4_distinct[k] < 5:
                b += (5 - k4_distinct[k])
        return b
        
    current_badness = calc_badness()
    current_palette = palette_size()
    best_colors = colors[:]
    best_badness = current_badness
    best_palette = current_palette
    
    # Count of each color in the whole graph
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
        
    def update_state(e, old_c, new_c):
        """Updates colors, k4_counts, k4_distinct, color_counts. Returns delta_badness."""
        nonlocal current_badness, current_palette, best_badness, best_palette, best_colors
        nonlocal color_counts
        
        # 1. Update global color counts
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            del color_counts[old_c]
            # Palette size decreases by 1
            current_palette -= 1
            
        color_counts[new_c] = color_counts.get(new_c, 0) + 1
        if color_counts[new_c] == 1:
            # Palette size increases by 1
            current_palette += 1
            
        # 2. Update K4 states
        delta_badness = 0
        for k in edge_to_k4s[e]:
            # Update counts
            cnt_old = k4_counts[k].get(old_c, 0)
            if cnt_old == 1:
                del k4_counts[k][old_c]
                k4_distinct[k] -= 1
            else:
                k4_counts[k][old_c] = cnt_old - 1
                
            cnt_new = k4_counts[k].get(new_c, 0)
            if cnt_new == 0:
                k4_counts[k][new_c] = 1
                k4_distinct[k] += 1
            else:
                k4_counts[k][new_c] = cnt_new + 1
                
            # Update badness for this K4
            old_d = k4_distinct[k]
            # Wait, I updated k4_distinct already. I need to compare before/after for badness.
            # Let's calculate badness contribution before and after.
            # Actually, I can just recalculate the badness contribution for this K4.
            # But I need the "old" badness contribution to subtract.
            # Let's store badness_contrib[k] = max(0, 5 - k4_distinct[k])
            # Then delta = new_contrib - old_contrib.
            
        # This approach is getting messy with side effects.
        # Let's write a cleaner move evaluation.
        
    # Restart with cleaner implementation
    
    # State
    colors = [0] * E
    for (i, j), k in edge_idx.items():
        colors[k] = (i + j) % 17
        
    k4_counts = [{} for _ in range(K4_COUNT)]
    k4_distinct = [0] * K4_COUNT
    k4_badness = [0] * K4_COUNT
    
    for k in range(K4_COUNT):
        for e in k4s[k]:
            c = colors[e]
            k4_counts[k][c] = k4_counts[k].get(c, 0) + 1
        k4_distinct[k] = len(k4_counts[k])
        k4_badness[k] = max(0, 5 - k4_distinct[k])
        
    total_badness = sum(k4_badness)
    
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    current_palette = len(color_counts)
    
    best_colors = colors[:]
    best_badness = total_badness
    best_palette = current_palette
    
    def apply_move(e, old_c, new_c):
        """Applies move and updates state. Returns delta_badness."""
        nonlocal total_badness, current_palette
        nonlocal color_counts, k4_counts, k4_distinct, k4_badness
        
        # 1. Global Color Counts
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            del color_counts[old_c]
            current_palette -= 1
            
        if new_c not in color_counts:
            color_counts[new_c] = 1
            current_palette += 1
        else:
            color_counts[new_c] += 1
            
        # 2. K4 Updates
        delta_bad = 0
        for k in edge_to_k4s[e]:
            # Update old color
            cnt_o = k4_counts[k].get(old_c, 0)
            if cnt_o == 1:
                del k4_counts[k][old_c]
                k4_distinct[k] -= 1
            else:
                k4_counts[k][old_c] = cnt_o - 1
                
            # Update new color
            cnt_n = k4_counts[k].get(new_c, 0)
            if cnt_n == 0:
                k4_counts[k][new_c] = 1
                k4_distinct[k] += 1
            else:
                k4_counts[k][new_c] = cnt_n + 1
                
            # Update Badness
            old_bad = k4_badness[k]
            new_bad = max(0, 5 - k4_distinct[k])
            k4_badness[k] = new_bad
            delta_bad += (new_bad - old_bad)
            
        total_badness += delta_bad
        return delta_bad

    def revert_move(e, old_c, new_c):
        """Reverts a move. Assumes state was modified by apply_move(e, old_c, new_c)."""
        nonlocal total_badness, current_palette
        nonlocal color_counts, k4_counts, k4_distinct, k4_badness
        
        # 1. Global Color Counts (Reverse)
        # current state has new_c, we want old_c
        color_counts[new_c] -= 1
        if color_counts[new_c] == 0:
            del color_counts[new_c]
            current_palette -= 1
            
        if old_c not in color_counts:
            color_counts[old_c] = 1
            current_palette += 1
        else:
            color_counts[old_c] += 1
            
        # 2. K4 Updates (Reverse)
        delta_bad = 0
        for k in edge_to_k4s[e]:
            # Reverse new color -> old color
            # Currently has new_c, want old_c
            
            # Remove new_c
            cnt_n = k4_counts[k].get(new_c, 0)
            if cnt_n == 1:
                del k4_counts[k][new_c]
                k4_distinct[k] -= 1
            else:
                k4_counts[k][new_c] = cnt_n - 1
                
            # Add old_c
            cnt_o = k4_counts[k].get(old_c, 0)
            if cnt_o == 0:
                k4_counts[k][old_c] = 1
                k4_distinct[k] += 1
            else:
                k4_counts[k][old_c] = cnt_o + 1
                
            # Update Badness
            old_bad = k4_badness[k]
            new_bad = max(0, 5 - k4_distinct[k])
            k4_badness[k] = new_bad
            delta_bad += (new_bad - old_bad)
            
        total_badness += delta_bad

    max_iter = 10000000
    iter_count = 0
    no_improve = 0
    
    while _remaining() > 5.0 and iter_count < max_iter:
        iter_count += 1
        
        if total_badness > 0:
            # Strategy: Fix violations
            # Pick a random edge involved in a violation?
            # Or just random edge.
            e = rng.randrange(E)
            old_c = colors[e]
            # Try random colors
            candidates = [rng.randint(0, 40) for _ in range(10)]
        else:
            # Strategy: Reduce palette
            # Pick a random edge
            e = rng.randrange(E)
            old_c = colors[e]
            # Try existing colors
            pal = list(color_counts.keys())
            candidates = [c for c in pal if c != old_c]
            if not candidates:
                continue
            # Bias: Prefer colors that are already present to avoid increasing palette
            # Actually, we WANT to reduce palette.
            # If we change old_c to new_c, and count(old_c) becomes 0, palette shrinks.
            # So we should prefer moves that eliminate old_c.
            # But we can only eliminate old_c if it's the last edge of that color.
            # So, if color_counts[old_c] == 1, we MUST change it to something else to reduce palette.
            # If color_counts[old_c] > 1, changing it doesn't reduce palette unless we eventually eliminate it.
            
            # Heuristic: If color_counts[old_c] == 1, this is a "critical" edge for palette reduction.
            # Let's prioritize edges where old_c count is 1.
            # For now, just random.
            rng.shuffle(candidates)
            
        best_move_c = old_c
        best_move_score = total_badness * 10000 + current_palette
        best_move_badness = total_badness
        best_move_palette = current_palette
        best_delta = 0
        
        for new_c in candidates:
            if new_c == old_c:
                continue
                
            # Apply
            apply_move(e, old_c, new_c)
            
            new_score = total_badness * 10000 + current_palette
            
            if new_score < best_move_score:
                best_move_score = new_score
                best_move_c = new_c
                best_move_badness = total_badness
                best_move_palette = current_palette
                best_delta = new_score - (old_c_score) # Need old score? No, we compare to current best
                
            # Revert
            revert_move(e, old_c, new_c)
            
        # Apply best move
        if best_move_c != old_c:
            apply_move(e, old_c, best_move_c)
            colors[e] = best_move_c
            no_improve = 0
            
            # Update best
            if best_move_badness < best_badness or (best_move_badness == best_badness and best_move_palette < best_palette):
                best_colors = colors[:]
                best_badness = best_move_badness
                best_palette = best_move_palette
        else:
            no_improve += 1
            if no_improve > 5000:
                # Perturb
                # Randomly change 5 edges
                for _ in range(5):
                    pe = rng.randrange(E)
                    pc = colors[pe]
                    nc = rng.randint(0, 40)
                    apply_move(pe, pc, nc)
                    colors[pe] = nc
                no_improve = 0
                best_colors = colors[:]
                best_badness = total_badness
                best_palette = current_palette

    # Final Remap
    used = sorted(set(best_colors))
    mapping = {c: idx for idx, c in enumerate(used)}
    final_colors = [mapping[c] for c in best_colors]
    
    return final_colors

# Helper to get old score for comparison in loop? 
# In the loop above, I used `old_c_score` which was undefined. 
# I should just compare `new_score` to `current_best_score` which is `total_badness * 10000 + current_palette` BEFORE the move.
# Let's fix the loop logic.

def construct():
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
        
    K4_COUNT = len(k4s)
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)

    colors = [0] * E
    for (i, j), k in edge_idx.items():
        colors[k] = (i + j) % 17
        
    k4_counts = [{} for _ in range(K4_COUNT)]
    k4_distinct = [0] * K4_COUNT
    k4_badness = [0] * K4_COUNT
    
    for k in range(K4_COUNT):
        for e in k4s[k]:
            c = colors[e]
            k4_counts[k][c] = k4_counts[k].get(c, 0) + 1
        k4_distinct[k] = len(k4_counts[k])
        k4_badness[k] = max(0, 5 - k4_distinct[k])
        
    total_badness = sum(k4_badness)
    
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    current_palette = len(color_counts)
    
    best_colors = colors[:]
    best_badness = total_badness
    best_palette = current_palette
    
    def apply_move(e, old_c, new_c):
        nonlocal total_badness, current_palette
        nonlocal color_counts, k4_counts, k4_distinct, k4_badness
        
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            del color_counts[old_c]
            current_palette -= 1
            
        if new_c not in color_counts:
            color_counts[new_c] = 1
            current_palette += 1
        else:
            color_counts[new_c] += 1
            
        delta_bad = 0
        for k in edge_to_k4s[e]:
            cnt_o = k4_counts[k].get(old_c, 0)
            if cnt_o == 1:
                del k4_counts[k][old_c]
                k4_distinct[k] -= 1
            else:
                k4_counts[k][old_c] = cnt_o - 1
                
            cnt_n = k4_counts[k].get(new_c, 0)
            if cnt_n == 0:
                k4_counts[k][new_c] = 1
                k4_distinct[k] += 1
            else:
                k4_counts[k][new_c] = cnt_n + 1
                
            old_bad = k4_badness[k]
            new_bad = max(0, 5 - k4_distinct[k])
            k4_badness[k] = new_bad
            delta_bad += (new_bad - old_bad)
            
        total_badness += delta_bad

    def revert_move(e, old_c, new_c):
        nonlocal total_badness, current_palette
        nonlocal color_counts, k4_counts, k4_distinct, k4_badness
        
        color_counts[new_c] -= 1
        if color_counts[new_c] == 0:
            del color_counts[new_c]
            current_palette -= 1
            
        if old_c not in color_counts:
            color_counts[old_c] = 1
            current_palette += 1
        else:
            color_counts[old_c] += 1
            
        delta_bad = 0
        for k in edge_to_k4s[e]:
            cnt_n = k4_counts[k].get(new_c, 0)
            if cnt_n == 1:
                del k4_counts[k][new_c]
                k4_distinct[k] -= 1
            else:
                k4_counts[k][new_c] = cnt_n - 1
                
            cnt_o = k4_counts[k].get(old_c, 0)
            if cnt_o == 0:
                k4_counts[k][old_c] = 1
                k4_distinct[k] += 1
            else:
                k4_counts[k][old_c] = cnt_o + 1
                
            old_bad = k4_badness[k]
            new_bad = max(0, 5 - k4_distinct[k])
            k4_badness[k] = new_bad
            delta_bad += (new_bad - old_bad)
            
        total_badness += delta_bad

    rng = random.Random(42)
    max_iter = 10000000
    iter_count = 0
    no_improve = 0
    
    while _remaining() > 5.0 and iter_count < max_iter:
        iter_count += 1
        
        current_score = total_badness * 10000 + current_palette
        
        if total_badness > 0:
            e = rng.randrange(E)
            old_c = colors[e]
            candidates = [rng.randint(0, 40) for _ in range(10)]
        else:
            e = rng.randrange(E)
            old_c = colors[e]
            pal = list(color_counts.keys())
            candidates = [c for c in pal if c != old_c]
            if not candidates:
                continue
            rng.shuffle(candidates)
            
        best_move_c = old_c
        best_move_score = current_score
        
        for new_c in candidates:
            if new_c == old_c:
                continue
                
            apply_move(e, old_c, new_c)
            new_score = total_badness * 10000 + current_palette
            
            if new_score < best_move_score:
                best_move_score = new_score
                best_move_c = new_c
                
            revert_move(e, old_c, new_c)
            
        if best_move_c != old_c:
            apply_move(e, old_c, best_move_c)
            colors[e] = best_move_c
            no_improve = 0
            
            if total_badness < best_badness or (total_badness == best_badness and current_palette < best_palette):
                best_colors = colors[:]
                best_badness = total_badness
                best_palette = current_palette
        else:
            no_improve += 1
            if no_improve > 10000:
                # Perturb
                for _ in range(5):
                    pe = rng.randrange(E)
                    pc = colors[pe]
                    nc = rng.randint(0, 40)
                    apply_move(pe, pc, nc)
                    colors[pe] = nc
                no_improve = 0
                best_colors = colors[:]
                best_badness = total_badness
                best_palette = current_palette

    used = sorted(set(best_colors))
    mapping = {c: idx for idx, c in enumerate(used)}
    final_colors = [mapping[c] for c in best_colors]
    
    return final_colors
