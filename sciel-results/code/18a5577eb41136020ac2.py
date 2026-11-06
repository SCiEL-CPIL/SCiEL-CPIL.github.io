"""Improved program for edge-colourings of K16 with >=5 colours per K4."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _idx(i, j):
    if i > j:
        i, j = j, i
    return i * 16 - i * (i + 1) // 2 + j - i - 1

def _count_bad_k4(colors_list):
    """Count number of K4s with fewer than 5 distinct colours."""
    bad = 0
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e1 = colors_list[_idx(a, b)]
        e2 = colors_list[_idx(a, c)]
        e3 = colors_list[_idx(a, d)]
        e4 = colors_list[_idx(b, c)]
        e5 = colors_list[_idx(b, d)]
        e6 = colors_list[_idx(c, d)]
        cols = {e1, e2, e3, e4, e5, e6}
        if len(cols) < 5:
            bad += 1
    return bad

def _make_17coloring():
    """The (i+j) % 17 construction - guaranteed valid, uses 17 colors."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _local_search_reduce_colors(colors, max_time, seed=42):
    """
    Local search to reduce the number of distinct colors used
    while maintaining the constraint that every K4 has >= 5 distinct colors.
    """
    rng = random.Random(seed)
    t0 = time.time()
    deadline = max_time
    
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for a, b, c, d in all_k4s:
        k4_edges.append([_idx(a,b), _idx(a,c), _idx(a,d), _idx(b,c), _idx(b,d), _idx(c,d)])
    
    edge_to_k4s = [[] for _ in range(120)]
    for ki, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(ki)
    
    def is_valid():
        for ki in range(len(all_k4s)):
            if len(set(colors[e] for e in k4_edges[ki])) < 5:
                return False
        return True
    
    if not is_valid():
        return colors, _count_bad_k4(colors), max(colors) + 1
    
    best_colors = colors[:]
    num_colors = len(set(colors))
    
    # Precompute color usage
    def get_color_counts():
        counts = {}
        for c in colors:
            counts[c] = counts.get(c, 0) + 1
        return counts
    
    while time.time() - t0 < deadline:
        counts = get_color_counts()
        if len(counts) <= 13:
            break
        
        # Pick the least-used color to try to eliminate
        target_color = min(counts.keys(), key=lambda c: counts[c])
        edges_with_target = [i for i, c in enumerate(colors) if c == target_color]
        
        if not edges_with_target:
            continue
        
        # Try to recolor each edge to a different existing color
        success = True
        for e in edges_with_target:
            found = False
            # Shuffle other colors for diversity
            other_colors = [c for c in counts.keys() if c != target_color]
            rng.shuffle(other_colors)
            for new_c in other_colors:
                colors[e] = new_c
                # Check all K4s containing this edge
                ok = True
                for ki in edge_to_k4s[e]:
                    if len(set(colors[ee] for ee in k4_edges[ki])) < 5:
                        ok = False
                        break
                if ok:
                    found = True
                    break
                else:
                    colors[e] = target_color
            if not found:
                success = False
                break
        
        if success:
            num_colors -= 1
            best_colors = colors[:]
    
    return best_colors, 0, num_colors

def _try_random_1factorizations(max_time, seeds):
    """Try multiple random 1-factorizations, return best valid one."""
    best = None
    best_bad = float('inf')
    
    for seed in seeds:
        t0 = time.time()
        if time.time() - t0 > max_time:
            break
        
        # Generate a random 1-factorization
        colors15 = [0] * 120
        perm = list(range(15))
        random.Random(seed).shuffle(perm)
        
        for i in range(15):
            colors15[_idx(15, perm[i])] = i
            for j in range(1, 8):
                a = perm[(i + j) % 15]
                b = perm[(i - j) % 15]
                if a > b:
                    a, b = b, a
                colors15[_idx(a, b)] = i
        
        bad15 = _count_bad_k4(colors15)
        if bad15 < best_bad:
            best_bad = bad15
            best = colors15[:]
        
        if bad15 == 0:
            return colors15
    
    return best, best_bad

def _fix_bad_k4s(colors, max_time, seed=42):
    """
    Try to fix bad K4s by recoloring edges.
    Used when we have a nearly-valid coloring.
    """
    rng = random.Random(seed)
    t0 = time.time()
    deadline = max_time
    
    all_k4s = list(itertools.combinations(range(16), 4))
    k4_edges = []
    for a, b, c, d in all_k4s:
        k4_edges.append([_idx(a,b), _idx(a,c), _idx(a,d), _idx(b,c), _idx(b,d), _idx(c,d)])
    
    edge_to_k4s = [[] for _ in range(120)]
    for ki, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(ki)
    
    bad_count = _count_bad_k4(colors)
    if bad_count == 0:
        return colors, 0
    
    best_colors = colors[:]
    best_bad = bad_count
    
    while time.time() - t0 < deadline:
        # Find bad K4s
        bad_kis = []
        for ki in range(len(all_k4s)):
            if len(set(colors[e] for e in k4_edges[ki])) < 5:
                bad_kis.append(ki)
        
        if not bad_kis:
            break
        
        # Pick a random bad K4
        ki = rng.choice(bad_kis)
        edges = k4_edges[ki]
        current_colors = [colors[e] for e in edges]
        current_distinct = len(set(current_colors))
        
        # Try to fix by recoloring one edge
        edge_choices = list(range(6))
        rng.shuffle(edge_choices)
        
        improved = False
        for ec in edge_choices:
            edge_idx = edges[ec]
            old_color = colors[edge_idx]
            
            # Try new colors: existing colors + new colors up to 14
            candidate_colors = list(set(colors)) | set(range(5, 15))
            rng.shuffle(candidate_colors)
            
            for new_color in candidate_colors:
                if new_color == old_color:
                    continue
                
                colors[edge_idx] = new_color
                
                # Check if this K4 is now good
                new_colors = [colors[e] for e in edges]
                if len(set(new_colors)) >= 5:
                    # Check if we made other K4s worse
                    # Only check K4s that contain this edge and were previously good
                    made_worse = 0
                    for other_ki in edge_to_k4s[edge_idx]:
                        if other_ki == ki:
                            continue
                        oc = [colors[e] for e in k4_edges[other_ki]]
                        if len(set(oc)) < 5:
                            made_worse += 1
                    
                    # Accept if we fixed the target and didn't make too many worse
                    if made_worse <= 1:
                        improved = True
                        break
                
                colors[edge_idx] = old_color
            
            if improved:
                break
        
        if improved:
            new_bad = _count_bad_k4(colors)
            if new_bad < best_bad:
                best_bad = new_bad
                best_colors = colors[:]
                if best_bad == 0:
                    break
    
    return best_colors, best_bad

def construct():
    t0 = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Start with the 17-coloring which is guaranteed valid
    colors17 = _make_17coloring()
    
    # Try to reduce color count from 17
    remaining = deadline - (time.time() - t0)
    if remaining > 20:
        result, bad, nc = _local_search_reduce_colors(colors17, min(80, remaining), seed=42)
        if bad == 0:
            return result
    
    # Try random 1-factorizations
    remaining = deadline - (time.time() - t0)
    if remaining > 10:
        seeds = [42, 123, 456, 789, 101112, 131415, 161718]
        best15, best_bad15 = _try_random_1factorizations(min(30, remaining), seeds)
        
        if best_bad15 == 0:
            return best15
        
        if best_bad15 < 50 and remaining > 10:
            result, bad = _fix_bad_k4s(best15, min(40, remaining), seed=999)
            if bad == 0:
                return result
    
    # Fallback: 17-coloring
    return colors17

# EVOLVE-BLOCK-END
