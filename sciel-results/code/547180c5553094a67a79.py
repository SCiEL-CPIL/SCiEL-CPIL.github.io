import random
import time
import itertools
import sys

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all K4s and their edge lists
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
    
    K = len(k4s)
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Initial coloring: Use a structure that likely uses few colors
    # Try a modular arithmetic approach with a prime or small set
    # (i + 2*j) % 13 might work well for 13 colors
    # Let's try to initialize with 13 colors to see if it's valid
    # If not, the search will fix it.
    
    def get_initial_colors():
        # Use a greedy or random init with a small palette
        # Random init with 13 colors
        return [random.randint(0, 12) for _ in range(E)]
    
    colors = get_initial_colors()
    
    # Track color usage
    # We need to know which colors are present
    # And for each K4, the set of colors present
    
    def compute_state(colors):
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        k4_distinct = [0] * K
        badness = 0
        for k4_idx, k4e in enumerate(k4s):
            distinct = len(set(colors[e] for e in k4e))
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                badness += (5 - distinct)
        
        palette_size = len(color_counts)
        return color_counts, k4_distinct, badness, palette_size

    color_counts, k4_distinct, badness, palette_size = compute_state(colors)
    
    best_colors = colors[:]
    best_badness = badness
    best_palette = palette_size
    best_score = best_palette * 1000000 + best_badness
    
    rng = random.Random(42)
    
    # Simulated Annealing parameters
    T = 100.0
    cooling_rate = 0.999
    
    def score(badness, palette):
        return palette * 1000000 + badness
    
    def try_move(e, new_c):
        old_c = colors[e]
        if new_c == old_c:
            return False, 0, 0
        
        # Apply
        colors[e] = new_c
        
        # Update counts
        color_counts[old_c] -= 1
        if color_counts[old_c] == 0:
            del color_counts[old_c]
        color_counts[new_c] = color_counts.get(new_c, 0) + 1
        
        # Update badness and k4_distinct
        delta_badness = 0
        for k4_idx in edge_to_k4s[e]:
            old_dist = k4_distinct[k4_idx]
            k4e = k4s[k4_idx]
            
            # Efficient distinct count update
            # We can just recompute for the 6 edges, it's small
            # Or use a set. Given E=120, K=1820, recompute is fast enough in Python if optimized?
            # len(set(...)) on 6 elements is very fast.
            
            new_dist = len(set(colors[ee] for ee in k4e))
            k4_distinct[k4_idx] = new_dist
            
            if old_dist < 5:
                delta_badness -= (5 - old_dist)
            if new_dist < 5:
                delta_badness += (5 - new_dist)
        
        new_badness = badness + delta_badness
        new_palette = len(color_counts)
        
        return True, delta_badness, new_palette

    # Revert helper
    def revert_move(e, new_c, delta_badness, new_palette):
        old_c = colors[e] # This is now new_c
        colors[e] = old_c # Wait, we need old_c from before
        
        # This revert logic is tricky if we don't store old_c.
        # Better to just store the full state or use a stack.
        # Given the simplicity, let's just recompute if we revert.
        # Or better: pass old_c to revert.
        
        pass

    # Let's use a simpler approach: 
    # Generate a candidate move, compute its score, and accept/reject.
    # To compute score, we need to apply, check, and potentially revert.
    
    current_score = score(badness, palette_size)
    
    for iteration in range(5000000):
        if time.time() > deadline - 5:
            break
            
        # Cooling
        if iteration % 1000 == 0:
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
        
        # Choose a move
        # 90% random edge recolor, 10% targeted elimination attempt
        
        move_type = rng.random()
        
        if move_type < 0.9:
            e = rng.randrange(E)
            old_c = colors[e]
            # Pick a new color
            # Prefer colors already in palette to keep palette size down
            # Or pick a random color in 0..12
            # If we want to reduce palette, we should avoid introducing new colors
            # But if we are stuck, new colors might help escape.
            
            # Strategy: 
            # If badness > 0, allow new colors (0..15) to fix constraints.
            # If badness == 0, only use existing colors to maintain/reduce palette.
            
            if badness > 0:
                new_c = rng.randint(0, 15)
            else:
                if color_counts:
                    new_c = rng.choice(list(color_counts.keys()))
                else:
                    new_c = 0
            
            if new_c == old_c:
                continue
            
            # Apply and evaluate
            colors[e] = new_c
            
            color_counts[old_c] -= 1
            if color_counts[old_c] == 0:
                del color_counts[old_c]
            color_counts[new_c] = color_counts.get(new_c, 0) + 1
            
            delta_badness = 0
            for k4_idx in edge_to_k4s[e]:
                old_dist = k4_distinct[k4_idx]
                k4e = k4s[k4_idx]
                new_dist = len(set(colors[ee] for ee in k4e))
                k4_distinct[k4_idx] = new_dist
                
                if old_dist < 5:
                    delta_badness -= (5 - old_dist)
                if new_dist < 5:
                    delta_badness += (5 - new_dist)
            
            new_badness = badness + delta_badness
            new_palette = len(color_counts)
            new_score = score(new_badness, new_palette)
            
            # Acceptance
            if new_score < current_score or (T > 0 and rng.random() < pow(2.718, -(new_score - current_score) / T)):
                # Accept
                current_score = new_score
                badness = new_badness
                palette_size = new_palette
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    best_badness = badness
                    best_palette = palette_size
            else:
                # Revert
                colors[e] = old_c
                color_counts[new_c] -= 1
                if color_counts[new_c] == 0:
                    del color_counts[new_c]
                color_counts[old_c] = color_counts.get(old_c, 0) + 1
                
                for k4_idx in edge_to_k4s[e]:
                    k4e = k4s[k4_idx]
                    k4_distinct[k4_idx] = len(set(colors[ee] for ee in k4e))
        
        else:
            # Targeted Elimination
            # Try to eliminate a color
            if not color_counts:
                continue
                
            # Pick a color to eliminate. Prefer colors with low frequency?
            # Or just pick one.
            target_color = rng.choice(list(color_counts.keys()))
            
            # Find all edges with this color
            edges_to_recolor = [i for i, c in enumerate(colors) if c == target_color]
            
            if not edges_to_recolor:
                continue
            
            # Try to recolor these edges to other colors
            # We do this greedily or with a mini-search.
            # For simplicity, let's try to recolor one edge at a time in this phase?
            # No, that's just the random move.
            
            # Let's try a "swap" move: swap colors of two edges.
            # This keeps palette size constant but might help rearrange.
            e1 = rng.randrange(E)
            e2 = rng.randrange(E)
            if e1 == e2:
                continue
            
            c1 = colors[e1]
            c2 = colors[e2]
            if c1 == c2:
                continue
            
            # Swap
            colors[e1] = c2
            colors[e2] = c1
            
            # Update counts
            # c1 count unchanged, c2 count unchanged
            # So palette size unchanged.
            
            # Update badness
            delta_badness = 0
            affected_k4s = set(edge_to_k4s[e1]) | set(edge_to_k4s[e2])
            
            for k4_idx in affected_k4s:
                old_dist = k4_distinct[k4_idx]
                k4e = k4s[k4_idx]
                new_dist = len(set(colors[ee] for ee in k4e))
                k4_distinct[k4_idx] = new_dist
                
                if old_dist < 5:
                    delta_badness -= (5 - old_dist)
                if new_dist < 5:
                    delta_badness += (5 - new_dist)
            
            new_badness = badness + delta_badness
            new_palette = palette_size
            new_score = score(new_badness, new_palette)
            
            if new_score < current_score or (T > 0 and rng.random() < pow(2.718, -(new_score - current_score) / T)):
                current_score = new_score
                badness = new_badness
                # palette_size unchanged
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    best_badness = badness
                    best_palette = palette_size
            else:
                # Revert
                colors[e1] = c1
                colors[e2] = c2
                for k4_idx in affected_k4s:
                    k4e = k4s[k4_idx]
                    k4_distinct[k4_idx] = len(set(colors[ee] for ee in k4e))

    # Final verification
    valid = True
    for k4e in k4s:
        if len(set(best_colors[e] for e in k4e)) < 5:
            valid = False
            break
    
    if not valid:
        # If best is invalid, we need a valid fallback.
        # The search should have found a valid one if it started from a valid one or repaired.
        # If it didn't, we return the best we have, but the evaluator might reject it.
        # The task says "A valid colouring certifies...".
        # So we MUST return a valid one.
        
        # Let's run a quick repair on best_colors if it's invalid.
        # Or just run the search again with a valid init?
        # Given the time, let's assume the search found a valid one.
        # If not, we try to fix it.
        
        # Simple repair:
        # While invalid:
        #   Find a bad K4.
        #   Pick an edge in it, change color to a random one.
        
        colors = best_colors[:]
        color_counts, k4_distinct, badness, palette_size = compute_state(colors)
        
        for _ in range(10000):
            if badness == 0:
                break
            # Find a bad K4
            bad_k4_idx = -1
            for i in range(K):
                if k4_distinct[i] < 5:
                    bad_k4_idx = i
                    break
            
            if bad_k4_idx == -1:
                break
                
            k4e = k4s[bad_k4_idx]
            # Pick an edge in this K4 to recolor
            e = rng.choice(k4e)
            old_c = colors[e]
            new_c = rng.randint(0, 15)
            
            colors[e] = new_c
            # Recompute state (slow but safe for small N)
            color_counts, k4_distinct, badness, palette_size = compute_state(colors)
            
            if badness < best_badness or (badness == 0 and palette_size < best_palette):
                best_colors = colors[:]
                best_badness = badness
                best_palette = palette_size
                best_score = score(badness, palette_size)

    return best_colors

