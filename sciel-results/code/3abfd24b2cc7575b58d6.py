import time
import random
import itertools

# Precomputation for speed
N = 16
edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
num_edges = len(edges)
edge_index = {e: idx for idx, e in enumerate(edges)}

# k4s: list of lists of edge indices
k4s = []
for quad in itertools.combinations(range(N), 4):
    e_idx = []
    for a, b in itertools.combinations(quad, 2):
        e_idx.append(edge_index[(a, b)])
    k4s.append(e_idx)

# edge_k4s: for each edge, list of k4 indices
edge_k4s = [[] for _ in range(num_edges)]
for k4_idx, e_idxs in enumerate(k4s):
    for e_idx in e_idxs:
        edge_k4s[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize with the known 17-color solution
    colors = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    best_colors = colors.copy()
    best_count = len(set(colors))
    
    current_colors = colors.copy()
    current_count = best_count
    
    # Precompute edge indices for faster access in validity check
    # We will use a list of lists for k4s to access edge indices directly
    
    def is_valid_move(e_idx, new_c):
        # Check all K4s containing edge e_idx
        for k4_idx in edge_k4s[e_idx]:
            # Get the 6 edges in this K4
            k4_e_idxs = k4s[k4_idx]
            # Calculate distinct colors
            # Optimization: If the K4 already has > 5 colors, it's safe unless we change a color that is unique?
            # No, if we change c1 to c2, and c2 is already there, count might drop.
            # We must check.
            
            # Fast path: if new_c == current_colors[e_idx], no change
            if new_c == current_colors[e_idx]:
                return True
            
            # Manual set construction for speed
            # There are only 6 edges
            c1 = current_colors[k4_e_idxs[0]] if k4_e_idxs[0] != e_idx else new_c
            c2 = current_colors[k4_e_idxs[1]] if k4_e_idxs[1] != e_idx else new_c
            c3 = current_colors[k4_e_idxs[2]] if k4_e_idxs[2] != e_idx else new_c
            c4 = current_colors[k4_e_idxs[3]] if k4_e_idxs[3] != e_idx else new_c
            c5 = current_colors[k4_e_idxs[4]] if k4_e_idxs[4] != e_idx else new_c
            c6 = current_colors[k4_e_idxs[5]] if k4_e_idxs[5] != e_idx else new_c
            
            # Check distinct count
            # Unrolled check for speed
            if c1 == c2:
                if c1 == c3:
                    if c1 == c4:
                        if c1 == c5:
                            if c1 == c6:
                                return False # 1 color
                        elif c1 == c6:
                            return False # 2 colors (c1, c5) - wait, c5!=c1, c6==c1. Colors: c1, c5. Count 2.
                        else:
                            # c1, c5, c6
                            if c5 == c6:
                                return False # 2 colors
                            else:
                                # 3 colors. Need 5.
                                return False
                    elif c1 == c5:
                        # c1, c4, c5(=c1), c6 -> c1, c4, c6
                        if c4 == c6:
                            return False
                        else:
                            return False # 3 colors
                    elif c1 == c6:
                        # c1, c4, c5, c6(=c1) -> c1, c4, c5
                        if c4 == c5:
                            return False
                        else:
                            return False # 3 colors
                    else:
                        # c1, c4, c5, c6 all distinct from c1 and each other?
                        # c1, c4, c5, c6. Plus c2,c3 which are c1.
                        # Colors: c1, c4, c5, c6. Count 4.
                        return False
                elif c1 == c4:
                    # c1, c2(=c1), c3, c4(=c1), c5, c6 -> c1, c3, c5, c6
                    if c3 == c5:
                        if c3 == c6:
                            return False
                        else:
                            return False
                    elif c3 == c6:
                        return False
                    elif c5 == c6:
                        return False
                    else:
                        return False # 4 colors
                elif c1 == c5:
                    # c1, c2(=c1), c3, c4, c5(=c1), c6 -> c1, c3, c4, c6
                    if c3 == c4:
                        if c3 == c6:
                            return False
                        else:
                            return False
                    elif c3 == c6:
                        return False
                    elif c4 == c6:
                        return False
                    else:
                        return False
                elif c1 == c6:
                    # c1, c2(=c1), c3, c4, c5, c6(=c1) -> c1, c3, c4, c5
                    if c3 == c4:
                        if c3 == c5:
                            return False
                        else:
                            return False
                    elif c3 == c5:
                        return False
                    elif c4 == c5:
                        return False
                    else:
                        return False
                else:
                    # c1, c2(=c1), c3, c4, c5, c6. c1 != c3,c4,c5,c6.
                    # Colors: c1, c3, c4, c5, c6. Count 5.
                    # Check if c3,c4,c5,c6 are distinct
                    if c3 == c4:
                        return False
                    if c3 == c5:
                        return False
                    if c3 == c6:
                        return False
                    if c4 == c5:
                        return False
                    if c4 == c6:
                        return False
                    if c5 == c6:
                        return False
                    return True
            else:
                # c1 != c2
                if c1 == c3:
                    # c1, c2, c3(=c1), c4, c5, c6 -> c1, c2, c4, c5, c6
                    if c1 == c4:
                        if c1 == c5:
                            if c1 == c6:
                                return False
                            else:
                                # c1, c2, c6. Count 3
                                return False
                        elif c1 == c6:
                            return False
                        else:
                            # c1, c2, c5, c6
                            if c5 == c6:
                                return False
                            else:
                                return False
                    elif c1 == c5:
                        if c1 == c6:
                            return False
                        else:
                            # c1, c2, c4, c6
                            if c4 == c6:
                                return False
                            else:
                                return False
                    elif c1 == c6:
                        # c1, c2, c4, c5
                        if c4 == c5:
                            return False
                        else:
                            return False
                    else:
                        # c1, c2, c4, c5, c6
                        if c4 == c5:
                            if c4 == c6:
                                return False
                            else:
                                return False
                        elif c4 == c6:
                            return False
                        elif c5 == c6:
                            return False
                        else:
                            return True
                elif c1 == c4:
                    # c1, c2, c3, c4(=c1), c5, c6 -> c1, c2, c3, c5, c6
                    if c1 == c5:
                        if c1 == c6:
                            return False
                        else:
                            # c1, c2, c3, c6
                            if c2 == c3:
                                return False
                            if c2 == c6:
                                return False
                            if c3 == c6:
                                return False
                            return False
                    elif c1 == c6:
                        # c1, c2, c3, c5
                        if c2 == c3:
                            return False
                        if c2 == c5:
                            return False
                        if c3 == c5:
                            return False
                        return False
                    else:
                        # c1, c2, c3, c5, c6
                        if c2 == c3:
                            if c2 == c5:
                                if c2 == c6:
                                    return False
                                else:
                                    return False
                            elif c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c5:
                            if c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c6:
                            return False
                        elif c3 == c5:
                            if c3 == c6:
                                return False
                            else:
                                return False
                        elif c3 == c6:
                            return False
                        elif c5 == c6:
                            return False
                        else:
                            return True
                elif c1 == c5:
                    # c1, c2, c3, c4, c5(=c1), c6 -> c1, c2, c3, c4, c6
                    if c1 == c6:
                        # c1, c2, c3, c4
                        if c2 == c3:
                            return False
                        if c2 == c4:
                            return False
                        if c3 == c4:
                            return False
                        return False
                    else:
                        # c1, c2, c3, c4, c6
                        if c2 == c3:
                            if c2 == c4:
                                if c2 == c6:
                                    return False
                                else:
                                    return False
                            elif c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c4:
                            if c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c6:
                            return False
                        elif c3 == c4:
                            if c3 == c6:
                                return False
                            else:
                                return False
                        elif c3 == c6:
                            return False
                        elif c4 == c6:
                            return False
                        else:
                            return True
                elif c1 == c6:
                    # c1, c2, c3, c4, c5, c6(=c1) -> c1, c2, c3, c4, c5
                    if c2 == c3:
                        if c2 == c4:
                            if c2 == c5:
                                return False
                            else:
                                return False
                        elif c2 == c5:
                            return False
                        else:
                            return False
                    elif c2 == c4:
                        if c2 == c5:
                            return False
                        else:
                            return False
                    elif c2 == c5:
                        return False
                    elif c3 == c4:
                        if c3 == c5:
                            return False
                        else:
                            return False
                    elif c3 == c5:
                        return False
                    elif c4 == c5:
                        return False
                    else:
                        return True
                else:
                    # c1, c2, c3, c4, c5, c6 all distinct from c1? No, c1!=c2, c1!=c3...
                    # Just check if any other pair matches
                    if c2 == c3:
                        if c2 == c4:
                            if c2 == c5:
                                if c2 == c6:
                                    return False
                                else:
                                    return False
                            elif c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c5:
                            if c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c6:
                            return False
                        else:
                            # c1, c2, c4, c5, c6
                            if c4 == c5:
                                if c4 == c6:
                                    return False
                                else:
                                    return False
                            elif c4 == c6:
                                return False
                            elif c5 == c6:
                                return False
                            else:
                                return True
                    elif c2 == c4:
                        if c2 == c5:
                            if c2 == c6:
                                return False
                            else:
                                return False
                        elif c2 == c6:
                            return False
                        else:
                            # c1, c2, c3, c5, c6
                            if c3 == c5:
                                if c3 == c6:
                                    return False
                                else:
                                    return False
                            elif c3 == c6:
                                return False
                            elif c5 == c6:
                                return False
                            else:
                                return True
                    elif c2 == c5:
                        if c2 == c6:
                            return False
                        else:
                            # c1, c2, c3, c4, c6
                            if c3 == c4:
                                if c3 == c6:
                                    return False
                                else:
                                    return False
                            elif c3 == c6:
                                return False
                            elif c4 == c6:
                                return False
                            else:
                                return True
                    elif c2 == c6:
                        # c1, c2, c3, c4, c5
                        if c3 == c4:
                            if c3 == c5:
                                return False
                            else:
                                return False
                        elif c3 == c5:
                            return False
                        elif c4 == c5:
                            return False
                        else:
                            return True
                    elif c3 == c4:
                        if c3 == c5:
                            if c3 == c6:
                                return False
                            else:
                                return False
                        elif c3 == c6:
                            return False
                        else:
                            # c1, c2, c3, c5, c6
                            if c5 == c6:
                                return False
                            else:
                                return True
                    elif c3 == c5:
                        if c3 == c6:
                            return False
                        else:
                            # c1, c2, c3, c4, c6
                            if c4 == c6:
                                return False
                            else:
                                return True
                    elif c3 == c6:
                        # c1, c2, c3, c4, c5
                        if c4 == c5:
                            return False
                        else:
                            return True
                    elif c4 == c5:
                        if c4 == c6:
                            return False
                        else:
                            # c1, c2, c3, c4, c6
                            return True
                    elif c4 == c6:
                        # c1, c2, c3, c4, c5
                        return True
                    elif c5 == c6:
                        # c1, c2, c3, c4, c5
                        return True
                    else:
                        return True

    # Main Loop
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        
        # Pick target color
        # Use existing colors to encourage merging
        existing_colors = list(set(current_colors))
        # Occasionally allow a new color? No, we want to minimize.
        c_new = random.choice(existing_colors)
        
        if c_new == current_colors[e_idx]:
            continue
            
        if is_valid_move(e_idx, c_new):
            # Accept move
            current_colors[e_idx] = c_new
            new_count = len(set(current_colors))
            
            if new_count < best_count:
                best_count = new_count
                best_colors = current_colors.copy()
                
            # If we found a very good solution (e.g., 13), we can stop early?
            # Theoretical min is 13.
            if best_count <= 13:
                break
        else:
            # Reject move (simulated annealing would accept with prob, but here we stick to valid moves only to ensure final validity)
            pass
            
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return best_colors
