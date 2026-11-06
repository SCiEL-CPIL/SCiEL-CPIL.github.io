import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initial colors: 1-factorization of K16
    # Vertices 0..14 and 15 (infinity)
    # Matchings M_k for k in 0..14
    # M_k contains (15, k) and ( (k+i)%15, (k-i)%15 ) for i=1..7
    colors = [0] * num_edges
    for k in range(15):
        # Edge (15, k)
        e_idx = edge_index[(min(k, 15), max(k, 15))]
        colors[e_idx] = k
        
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u > v:
                u, v = v, u
            e_idx = edge_index[(u, v)]
            colors[e_idx] = k

    # Verify seed is valid
    seed_valid = True
    for k4_idx in range(num_k4s):
        e_list = k4_edges[k4_idx]
        if len({colors[e] for e in e_list}) < 5:
            seed_valid = False
            break
    
    if not seed_valid:
        # Fallback to (u+v)%17
        for i, (u, v) in enumerate(edges):
            colors[i] = (u + v) % 17

    best_colors = colors[:]
    best_num_colors = max(colors) + 1
    
    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    def can_merge(c_from, c_to, colors):
        """
        Check if all edges of color c_from can be changed to c_to.
        Returns True if safe.
        """
        # Find all edges with color c_from
        edges_to_change = [e for e in range(num_edges) if colors[e] == c_from]
        
        for e_idx in edges_to_change:
            for k4_idx in edge_to_k4s[e_idx]:
                e_list = k4_edges[k4_idx]
                # Get colors of other 5 edges
                other_colors = set()
                for e in e_list:
                    if e != e_idx:
                        other_colors.add(colors[e])
                
                # Check condition:
                # If |other_colors| == 4 and c_to in other_colors, it's unsafe.
                # Why?
                # Original K4 colors: other_colors U {c_from}
                # Since c_from is unique in this K4 (if |other_colors|=4, and c_from not in other_colors, size 5. If c_from in other_colors, size 4? No, if c_from in other_colors, size of other_colors is at least 1. If |other_colors|=4, and c_from is one of them, then original size is 4. But we require >=5. So if |other_colors|=4, c_from MUST NOT be in other_colors for the original to be valid (size 5).
                # So if |other_colors|=4, c_from is NOT in other_colors.
                # New colors: other_colors U {c_to}.
                # If c_to in other_colors, new size = 4. Invalid.
                # If c_to not in other_colors, new size = 5. Valid.
                
                if len(other_colors) == 4 and c_to in other_colors:
                    return False
        return True

    # Local Search Loop using Merges
    # We maintain a list of current colors
    # We try to merge random pairs of colors
    
    iter_count = 0
    no_improve_count = 0
    current_colors = colors[:]
    
    while time.time() < deadline:
        # Compact current colors to keep indices small
        current_colors = compact_colors(current_colors)
        num_unique = max(current_colors) + 1 if current_colors else 0
        
        if num_unique < best_num_colors:
            best_num_colors = num_unique
            best_colors = current_colors[:]
        
        if best_num_colors == 13:
            break # Hit lower bound? Or just stop if we get very low.
            
        improved = False
        # Try to merge
        if num_unique > 1:
            # Get list of colors in use
            colors_in_use = list(set(current_colors))
            random.shuffle(colors_in_use)
            
            # Try a few random merges
            for _ in range(50):
                if len(colors_in_use) < 2:
                    break
                c_from = random.choice(colors_in_use)
                c_to = random.choice(colors_in_use)
                if c_from == c_to:
                    continue
                
                if can_merge(c_from, c_to, current_colors):
                    # Perform merge
                    for e in range(num_edges):
                        if current_colors[e] == c_from:
                            current_colors[e] = c_to
                    improved = True
                    # Remove c_from from colors_in_use? 
                    # It's safer to re-evaluate colors_in_use after compacting.
                    # But for the loop, we can just break and compact.
                    break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # Kick: If no merge works, try to change a single edge to a "new" color 
                # or a random existing color to break symmetry?
                # Actually, if no merge works, we might be stuck.
                # We can try to "split" a color: pick an edge of color C, change it to C+1.
                # This increases color count, but might allow subsequent merges.
                
                # Let's try a random edge change to a new color (num_unique)
                e_idx = random.randint(0, num_edges - 1)
                new_color = num_unique
                # Check if this single change is valid
                # For every K4 containing e_idx:
                # Old colors S. New colors (S \ {old}) U {new}.
                # We need size >= 5.
                
                valid_kick = True
                old_c = current_colors[e_idx]
                for k4_idx in edge_to_k4s[e_idx]:
                    e_list = k4_edges[k4_idx]
                    other_colors = set()
                    for e in e_list:
                        if e != e_idx:
                            other_colors.add(current_colors[e])
                    
                    # Original size: len(other_colors U {old_c})
                    # New size: len(other_colors U {new_color})
                    # Since new_color is new (not in other_colors, and not old_c),
                    # New size = len(other_colors) + 1
                    # We need len(other_colors) + 1 >= 5 => len(other_colors) >= 4.
                    
                    if len(other_colors) < 4:
                        valid_kick = False
                        break
                
                if valid_kick:
                    current_colors[e_idx] = new_color
                    no_improve_count = 0 # Reset? Or keep? 
                    # This is an expansion, so we don't improve best, but we escape local min.
        
        iter_count += 1
        if iter_count % 1000 == 0:
             # Periodic check
             pass

    # Final compact best
    final_compact = compact_colors(best_colors)
    return final_compact
