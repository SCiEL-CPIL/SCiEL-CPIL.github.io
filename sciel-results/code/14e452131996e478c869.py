import time
import random
import numpy as np

N = 9
NUM_VERTICES = 1 << N
TIME_LIMIT = 115.0  # Safety margin

# Precompute neighbors for all vertices
NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    best_path = []
    
    # Symmetry: Fix start at 0, first move bit 0, second move bit 1
    # Path: 0 -> 1 -> 3
    # We can vary the order of subsequent bits or use different starting triples if we break symmetry more, 
    # but 0-1-3 is a canonical start.
    
    # To explore different branches, we can randomize the order in which we consider bit flips 
    # or use different initial "blocked" states if we didn't fix the prefix.
    # Since we fix 0->1->3, the variation comes from the beam search tie-breaking.
    
    beam_width = 200
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Initialize beam with the fixed prefix
        # Path: [0, 1, 3]
        path = [0, 1, 3]
        used = {0, 1, 3}
        # blocked: neighbors of v0 and v1
        blocked = set(NEIGHBORS[0])
        blocked.update(NEIGHBORS[1])
        
        # Beam list of (path, used, blocked, score)
        # Score for heuristic: number of valid moves at next step
        def get_score(p, u, b):
            # Count valid moves from p[-1]
            count = 0
            for nb in NEIGHBORS[p[-1]]:
                if nb not in u and nb not in b:
                    count += 1
            return count
        
        current_score = get_score(path, used, blocked)
        beam = [(path, used, blocked, current_score)]
        
        while True:
            if time.time() - start_time > TIME_LIMIT:
                if len(path) > len(best_path):
                    best_path = path
                break
            
            next_beam = []
            for p, u, b, s in beam:
                cur = p[-1]
                for nb in NEIGHBORS[cur]:
                    if nb in u or nb in b:
                        continue
                    # Valid move
                    new_p = p + [nb]
                    new_u = u | {nb}
                    new_b = b | set(NEIGHBORS[p[-2]]) # Add neighbors of the second-to-last vertex
                    # Note: p[-2] is the vertex that was last in the previous step.
                    # In the new path, p[-2] is now at index len(new_p)-2.
                    # The constraint for the NEXT step will be against neighbors of new_p[0..-2].
                    # The previous 'b' contained neighbors of p[0..-2].
                    # We need to add neighbors of p[-1] (which is cur) to form the new 'b' for the next iteration?
                    # Let's re-verify the 'blocked' logic.
                    
                    # At step k, we have path P_k = [v_0, ..., v_k].
                    # We are finding v_{k+1}.
                    # v_{k+1} must not be neighbor of v_0, ..., v_{k-1}.
                    # So 'blocked' for step k should contain neighbors of v_0, ..., v_{k-1}.
                    
                    # In the loop above:
                    # We have path p = [v_0, ..., v_k].
                    # 'b' contains neighbors of v_0, ..., v_{k-1}.
                    # We pick nb = v_{k+1}.
                    # New path new_p = [v_0, ..., v_k, v_{k+1}].
                    # For the NEXT step (finding v_{k+2}), we need to check against neighbors of v_0, ..., v_k.
                    # So new_b should be b U neighbors(v_k).
                    # v_k is p[-1].
                    
                    new_b = b | set(NEIGHBORS[p[-1]])
                    
                    new_s = get_score(new_p, new_u, new_b)
                    next_beam.append((new_p, new_u, new_b, new_s))
                    
            if not next_beam:
                break
            
            # Sort by score descending, take top beam_width
            next_beam.sort(key=lambda x: x[3], reverse=True)
            # Randomize ties to explore different branches
            top_scores = next_beam[0][3]
            candidates = [x for x in next_beam if x[3] >= top_scores - 1]
            if len(candidates) > beam_width:
                random.shuffle(candidates)
                beam = candidates[:beam_width]
            else:
                beam = candidates
            
            # Update best path
            if len(beam[0][0]) > len(best_path):
                best_path = beam[0][0]
                
            # If no path can be extended, break
            if all(x[3] == 0 for x in beam):
                break
                
    return best_path
