import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute cycles
    # For each edge, store list of tuples of the other 3 edge indices in that cycle
    # This allows fast validity check for adding a NEW color.
    # If we add a new color to edge e, it's a rainbow cycle iff the other 3 edges have 3 distinct colors.
    
    edge_to_other_edges = [[] for _ in range(NUM_EDGES)]
    
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][c]
        e3 = edge_idx[c][d]
        e4 = edge_idx[d][a]
        
        edge_to_other_edges[e1].append((e2, e3, e4))
        edge_to_other_edges[e2].append((e1, e3, e4))
        edge_to_other_edges[e3].append((e1, e2, e4))
        edge_to_other_edges[e4].append((e1, e2, e3))
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][d]
        e3 = edge_idx[d][c]
        e4 = edge_idx[c][a]
        
        edge_to_other_edges[e1].append((e2, e3, e4))
        edge_to_other_edges[e2].append((e1, e3, e4))
        edge_to_other_edges[e3].append((e1, e2, e4))
        edge_to_other_edges[e4].append((e1, e2, e3))
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_idx[a][c]
        e2 = edge_idx[c][b]
        e3 = edge_idx[b][d]
        e4 = edge_idx[d][a]
        
        edge_to_other_edges[e1].append((e2, e3, e4))
        edge_to_other_edges[e2].append((e1, e3, e4))
        edge_to_other_edges[e3].append((e1, e2, e4))
        edge_to_other_edges[e4].append((e1, e2, e3))

    # Initial coloring: c(i,j) = i
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx[i][j]] = i
            
    freq = [0] * MAX_COLORS
    for c in colors:
        freq[c] += 1
        
    used_colors = set(colors)
    unused_colors = [c for c in range(MAX_COLORS) if c not in used_colors]
    
    best_score = len(used_colors)
    best_colors = colors.copy()
    
    random.seed(42)
    
    def is_valid_new_color(e, new_color):
        """
        Checks if assigning new_color (which is currently UNUSED) to edge e
        creates any rainbow 4-cycles.
        Since new_color is new, it is distinct from all other colors.
        A cycle becomes rainbow if the other 3 edges have 3 distinct colors.
        """
        for o1, o2, o3 in edge_to_other_edges[e]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            if c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def is_valid_any_color(e, new_color, old_color):
        """
        Checks if assigning new_color to edge e (replacing old_color)
        creates any rainbow 4-cycles.
        Used for shuffling.
        """
        if new_color == old_color:
            return True
        for o1, o2, o3 in edge_to_other_edges[e]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            # Check if {new_color, c1, c2, c3} are all distinct
            if new_color != c1 and new_color != c2 and new_color != c3:
                if c1 != c2 and c1 != c3 and c2 != c3:
                    return False
        return True

    while time.time() < deadline:
        # Try to add a new color
        move_found = False
        # Prioritize edges with high frequency old colors
        # Sort edges by freq[colors[e]] descending? 
        # Just iterate randomly or sorted.
        # To maximize score gain, we want to pick an edge where freq[old] > 1.
        
        # Let's try all unused colors and all edges
        # Optimize: Pick a random unused color, then scan edges?
        # Or pick random edge, then check all unused?
        
        # Heuristic: Try to find an edge that is "safe" for many new colors.
        # For now, simple scan.
        
        # To speed up, we can shuffle unused_colors and edges
        random.shuffle(unused_colors)
        # Don't shuffle edges every time, just iterate.
        
        for c_new in unused_colors:
            if time.time() > deadline:
                break
            for e in range(NUM_EDGES):
                if time.time() > deadline:
                    break
                old_c = colors[e]
                # Only accept if we increase score (freq[old_c] > 1)
                # Or if we are desperate and accept score-neutral moves?
                # If freq[old_c] == 1, score stays same.
                # If freq[old_c] > 1, score increases.
                
                # Always try to increase score first.
                if freq[old_c] > 1:
                    if is_valid_new_color(e, c_new):
                        # Accept
                        colors[e] = c_new
                        freq[old_c] -= 1
                        freq[c_new] += 1
                        used_colors.add(c_new)
                        if freq[old_c] == 0:
                            used_colors.discard(old_c)
                            unused_colors.append(old_c)
                        unused_colors.remove(c_new)
                        
                        current_score = len(used_colors)
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = colors.copy()
                        
                        move_found = True
                        break
            if move_found:
                break
        
        if not move_found:
            # No score-increasing move found.
            # Try a score-neutral move (shuffle) to escape local minima.
            # Pick random edge, random used color, check validity.
            e = random.randint(0, NUM_EDGES - 1)
            old_c = colors[e]
            # Pick a random different used color
            other_used = list(used_colors)
            if len(other_used) > 1:
                c_new = random.choice(other_used)
                if c_new != old_c:
                    if is_valid_any_color(e, c_new, old_c):
                        colors[e] = c_new
                        freq[old_c] -= 1
                        freq[c_new] += 1
                        if freq[old_c] == 0:
                            used_colors.discard(old_c)
                            unused_colors.append(old_c)
                        if freq[c_new] == 1: # It was already used, so freq was >=1, now >=2? 
                             # No, if it was used, it's in used_colors.
                             # If freq becomes 0, it leaves used_colors.
                             pass 
                        # Note: c_new was already in used_colors, so it stays.
                        # We don't update best_score here as it's neutral.

    return best_colors
