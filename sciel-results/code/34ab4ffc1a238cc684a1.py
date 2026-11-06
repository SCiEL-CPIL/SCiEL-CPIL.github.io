import random
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
# Precompute position matrix
pos = [[0]*N for _ in range(N)]
for i, (u, v) in enumerate(P['edge_order']):
    pos[u][v] = i + 1
    pos[v][u] = i + 1

def get_cycle_edges(cycle):
    edges = []
    for i in range(N):
        u = cycle[i]
        v = cycle[(i+1) % N]
        edges.append((u, v))
    return edges

def calc_max_score(cycle):
    max_score = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i+1) % N]
        if pos[u][v] > max_score:
            max_score = pos[u][v]
    return max_score

def two_opt_improve(cycle, max_score):
    """
    Attempts to improve the cycle by reducing the max score.
    Returns (new_cycle, new_max_score, improved)
    """
    # Build weight counts
    weights = [0] * (N*(N-1)//2 + 1)
    edge_set = set()
    for i in range(N):
        u = cycle[i]
        v = cycle[(i+1) % N]
        w = pos[u][v]
        weights[w] += 1
        # Normalize edge for set
        if u > v: u, v = v, u
        edge_set.add((u, v))
    
    current_max = max_score
    improved = False
    new_cycle = cycle[:]
    
    # Try 2-opt moves
    # We only care about moves that break edges with weight == current_max
    # and add edges with weight < current_max
    
    # Identify edges with current_max
    max_edges = []
    for i in range(N):
        u = new_cycle[i]
        v = new_cycle[(i+1) % N]
        if pos[u][v] == current_max:
            max_edges.append(i)
            
    if not max_edges:
        return new_cycle, current_max, False

    # Shuffle max edges to randomize search
    random.shuffle(max_edges)
    
    for i in max_edges:
        # Edge i is (new_cycle[i], new_cycle[i+1])
        # 2-opt: reverse segment from i to j
        # This breaks edge (i-1, i) and (j, j+1)
        # Wait, standard 2-opt on cycle:
        # Cut (u1, u2) and (u3, u4), reconnect (u1, u3) and (u2, u4)
        # In a list representation, reversing [i..j] breaks (i-1, i) and (j, j+1)
        # and adds (i-1, j) and (i, j+1)
        
        # Let's iterate over all j > i+1
        for j in range(i+1, N):
            if j == (i+N-1)%N: continue # adjacent, no change
            
            # Edges to remove:
            u1 = new_cycle[i-1]
            u2 = new_cycle[i]
            u3 = new_cycle[j]
            u4 = new_cycle[(j+1)%N]
            
            w1 = pos[u1][u2]
            w2 = pos[u3][u4]
            
            # Edges to add:
            u1_new = new_cycle[i-1]
            u2_new = new_cycle[j]
            u3_new = new_cycle[i]
            u4_new = new_cycle[(j+1)%N]
            
            w3 = pos[u1_new][u2_new]
            w4 = pos[u3_new][u4_new]
            
            # Check if this move breaks a max edge
            # The move breaks edges at indices i-1 and j (in the cycle order)
            # Specifically, it removes (u1, u2) which is edge i-1? 
            # Let's trace indices carefully.
            # Cycle: ... u_{i-1} - u_i ... u_j - u_{j+1} ...
            # Reverse u_i ... u_j
            # New: ... u_{i-1} - u_j ... u_i - u_{j+1} ...
            # Removed: (u_{i-1}, u_i) and (u_j, u_{j+1})
            # Added: (u_{i-1}, u_j) and (u_i, u_{j+1})
            
            # We are iterating i from max_edges.
            # max_edges contains indices k such that edge (cycle[k], cycle[k+1]) has max weight.
            # If we choose i from max_edges, we are targeting the edge starting at i.
            # But 2-opt reversal [i..j] removes edge (i-1, i) and (j, j+1).
            # So if we want to break edge i, we should consider reversals that include edge i.
            # Edge i is (cycle[i], cycle[i+1]).
            # Reversal [i+1 .. j] removes (i, i+1) and (j, j+1).
            # Let's adjust: If we pick a max edge at index k, we can try reversing [k+1 .. j] or [i .. k].
            
            # Simpler: Just try all 2-opt moves and check if they reduce max.
            # But that's O(N^2) checks. N=24, 276 moves. Very fast.
            
            # Let's just do all 2-opt moves in one pass and pick the best one.
            pass

    # Let's restart with a simpler, exhaustive 2-opt pass
    best_move = None
    best_new_max = current_max
    best_new_cycle = None
    
    for i in range(N):
        for j in range(i+1, N):
            if j == (i+N-1)%N: continue
            
            u1 = new_cycle[i-1]
            u2 = new_cycle[i]
            u3 = new_cycle[j]
            u4 = new_cycle[(j+1)%N]
            
            w1 = pos[u1][u2]
            w2 = pos[u3][u4]
            
            w3 = pos[u1][u3]
            w4 = pos[u2][u4]
            
            # Calculate new max
            # We need to know if w1 or w2 was the unique max, or if there are others.
            # Efficient way: 
            # If w3 > current_max or w4 > current_max, this move is bad (increases max).
            if w3 > current_max or w4 > current_max:
                continue
                
            # If w1 == current_max or w2 == current_max, we might reduce max.
            # If neither was current_max, max stays current_max (unless w3/w4 are lower, but max is determined by others).
            # Actually, if neither w1 nor w2 is current_max, and w3,w4 < current_max, the max remains current_max (from other edges).
            # So we only get a strict improvement if we remove ALL edges with current_max.
            
            # Let's count how many edges have current_max.
            # This is expensive to do inside the loop.
            # Precompute max_count.
            
            # Optimization: 
            # If w1 < current_max and w2 < current_max:
            #   New max is still current_max (assuming other edges exist). No improvement in max.
            #   But might help future moves? Yes.
            # If w1 == current_max or w2 == current_max:
            #   We removed a max edge.
            #   If w3 < current_max and w4 < current_max:
            #       New max is max(current_max - 1, max_other_edges).
            #       We need to check if any other edge has current_max.
            
            # Let's just calculate the new max explicitly for valid candidates.
            # Candidate is valid if w3 <= current_max and w4 <= current_max.
            
            # To be efficient, we only evaluate new_max if we removed at least one current_max edge.
            if w1 < current_max and w2 < current_max:
                continue # No max edge removed, max cannot decrease. (It can stay same, but we prioritize decreasing)
                
            # Construct new cycle to check max? 
            # Or just check if any OTHER edge has current_max.
            # If no other edge has current_max, and w3,w4 < current_max, then new max < current_max.
            # If another edge has current_max, new max == current_max.
            
            # Count of current_max in new_cycle:
            # Original count C.
            # Removed w1, w2. Added w3, w4.
            # New count C' = C - (1 if w1==M else 0) - (1 if w2==M else 0) + (1 if w3==M else 0) + (1 if w4==M else 0)
            
            # We need C to start.
            # Let's precompute C for the current cycle.
            
    # This logic is getting complex. Let's just do a simple hill climbing with full recalculation of max for N=24.
    # N=24 is small enough that recalculating max (24 lookups) is fast.
    
    return new_cycle, current_max, False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_cycle = list(range(N))
    best_score = calc_max_score(best_cycle)
    
    # Local search
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize cycle
        if restart == 0:
            cycle = best_cycle[:]
        else:
            cycle = list(range(N))
            random.shuffle(cycle)
            
        current_score = calc_max_score(cycle)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # 2-opt
            for i in range(N):
                if time.time() > deadline: break
                for j in range(i+1, N):
                    if j == (i+N-1)%N: continue
                    
                    u1 = cycle[i-1]
                    u2 = cycle[i]
                    u3 = cycle[j]
                    u4 = cycle[(j+1)%N]
                    
                    w1 = pos[u1][u2]
                    w2 = pos[u3][u4]
                    w3 = pos[u1][u3]
                    w4 = pos[u2][u4]
                    
                    # Quick reject if new edges are worse than current max
                    if w3 > current_score or w4 > current_score:
                        continue
                    
                    # Check if this move improves the max
                    # Only if we removed an edge with weight == current_score
                    if w1 < current_score and w2 < current_score:
                        continue
                        
                    # Apply move
                    cycle[i:j+1] = reversed(cycle[i:j+1])
                    
                    new_score = calc_max_score(cycle)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        cycle[i:j+1] = reversed(cycle[i:j+1])
                        
        if current_score < best_score:
            best_score = current_score
            best_cycle = cycle[:]
            
    return best_cycle
# EVOLVE-BLOCK-END
