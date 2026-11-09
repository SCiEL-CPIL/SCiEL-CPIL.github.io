import time
import random
import numpy as np

N = 9
MAX_TIME = 118.0
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    
    # Precompute neighbors as a numpy array for fast access
    neighbors = np.empty((NUM_VERTICES, N), dtype=np.uint32)
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    best_path = None
    best_len = 0
    
    rng = np.random.default_rng(seed=42)
    
    def get_valid_moves(v, used, forbidden):
        """Get list of valid neighbors for vertex v."""
        valid = []
        for n in neighbors[v]:
            if not used[n] and not forbidden[n]:
                valid.append(n)
        return valid

    def try_bidirectional(start_v):
        """
        Try to build a snake starting from start_v using bidirectional growth.
        """
        # Pick a random neighbor to form initial edge
        nbrs = neighbors[start_v]
        init_next = nbrs[rng.integers(N)]
        
        path = [start_v, init_next]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        used[init_next] = True
        
        # forbidden: vertices adjacent to internal vertices.
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        
        head = init_next
        tail = start_v
        
        while True:
            # Check time
            if time.time() - start_time > MAX_TIME:
                break
            
            # Get valid moves for head and tail
            head_moves = get_valid_moves(head, used, forbidden)
            tail_moves = get_valid_moves(tail, used, forbidden)
            
            if not head_moves and not tail_moves:
                break
            
            # Decide which end to extend
            # Warnsdorff: extend the more constrained end
            if len(head_moves) < len(tail_moves):
                extend_head = True
            elif len(tail_moves) < len(head_moves):
                extend_head = False
            else:
                extend_head = rng.random() < 0.5
            
            if extend_head:
                moves = head_moves
                if not moves:
                    break
                # Choose best move: minimize future options for the new head
                best_score = 100
                best_cand = moves[0]
                
                for c in moves:
                    score = 0
                    for nc in neighbors[c]:
                        if used[nc]:
                            continue
                        if forbidden[nc]:
                            continue
                        # If nc is neighbor of head (which becomes internal) and nc != c, it becomes forbidden
                        if (nc ^ head).bit_count() == 1:
                            if nc != c:
                                continue
                        score += 1
                    
                    if score < best_score:
                        best_score = score
                        best_cand = c
                    
                # Execute move
                path.append(best_cand)
                used[best_cand] = True
                
                # Update forbidden: head becomes internal
                for nh in neighbors[head]:
                    if nh != best_cand and not used[nh]:
                        forbidden[nh] = True
                
                head = best_cand
                
            else:
                moves = tail_moves
                if not moves:
                    break
                # Choose best move: minimize future options for the new tail
                best_score = 100
                best_cand = moves[0]
                
                for c in moves:
                    score = 0
                    for nc in neighbors[c]:
                        if used[nc]:
                            continue
                        if forbidden[nc]:
                            continue
                        # If nc is neighbor of tail (which becomes internal) and nc != c, it becomes forbidden
                        if (nc ^ tail).bit_count() == 1:
                            if nc != c:
                                continue
                        score += 1
                    
                    if score < best_score:
                        best_score = score
                        best_cand = c
                
                # Execute move
                path.insert(0, best_cand)
                used[best_cand] = True
                
                # Update forbidden: tail becomes internal
                for nt in neighbors[tail]:
                    if nt != best_cand and not used[nt]:
                        forbidden[nt] = True
                
                tail = best_cand
                
        return path

    # Run multiple restarts
    while time.time() - start_time < MAX_TIME:
        start_v = rng.integers(NUM_VERTICES)
        path = try_bidirectional(start_v)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    if best_path is None:
        # Fallback
        best_path = [0]
    
    # Convert numpy ints to python ints for safety
    return [int(v) for v in best_path]
