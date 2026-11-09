import time
import itertools
import numpy as np
from collections import deque

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Precompute all unique rotations for each box
    all_rotations = []
    for w, h, d in boxes:
        rots = list(set(itertools.permutations((w, h, d))))
        all_rotations.append(rots)
    
    # Strategy: Multiple diverse greedy attempts with different scoring heuristics
    # Keep the best result across all attempts
    
    best_score_overall = 0.0
    best_result_overall = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Different sorting strategies
    sort_strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume desc
        lambda i: boxes[i][0] * boxes[i][1],  # Base area desc
        lambda i: max(boxes[i]),  # Max dimension desc
        lambda i: min(boxes[i]),  # Min dimension asc (smaller first to fill gaps)
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
    ]
    
    # Different scoring functions for placement
    def make_scorer(mode):
        if mode == 0:  # Bottom-Left-Back (original)
            def score(cx, cy, cz):
                return cz * 1000000 + cy * 1000 + cx
            return score
        elif mode == 1:  # Prefer lower Z, then more central X,Y
            def score(cx, cy, cz):
                return cz * 1000000 + abs(W/2 - cx - W/4) * 100 + abs(H/2 - cy - H/4)
            return score
        elif mode == 2:  # Prefer filling corners
            def score(cx, cy, cz):
                corner_dist = (cx**2 + cy**2 + cz**2)
                return cz * 500000 + corner_dist
            return score
        elif mode == 3:  # Balance: lower Z, but also try to minimize max dimension usage
            def score(cx, cy, cz):
                return cz * 1000000 + cx + cy * 100
            return score
        else:  # mode 4: Prefer positions that maximize remaining space uniformity
            def score(cx, cy, cz):
                return cz * 800000 + cx * 2 + cy
            return score
    
    scorer_modes = [0, 1, 2, 3, 4]
    
    attempt_count = 0
    for sort_idx, sort_func in enumerate(sort_strategies):
        for scorer_mode in scorer_modes:
            if time.time() > deadline:
                break
            
            # Sort boxes using this strategy
            order = list(range(n))
            # For some strategies, we want descending; for others ascending
            if sort_idx == 3:  # Min dimension - ascending
                order.sort(key=sort_func)
            else:
                order.sort(key=sort_func, reverse=True)
            
            # Run greedy placement with this sorting and scoring
            placed = []
            candidates = set([(0, 0, 0)])
            results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
            scorer = make_scorer(scorer_mode)
            
            for idx in order:
                if time.time() > deadline:
                    break
                
                w, h, d = boxes[idx]
                best_pos = None
                best_rot = None
                best_score = float('inf')
                
                for rw, rh, rd in all_rotations[idx]:
                    for cx, cy, cz in candidates:
                        # Check bounds
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        
                        # Check overlap
                        valid = True
                        for px, py, pz, pw, ph, pd in placed:
                            if (cx < px + pw and px < cx + rw and
                                cy < py + ph and py < cy + rh and
                                cz < pz + pd and pz < cz + rd):
                                valid = False
                                break
                        
                        if valid:
                            s = scorer(cx, cy, cz)
                            if s < best_score:
                                best_score = s
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                
                if best_pos is not None:
                    bx, by, bz = best_pos
                    bw, bh, bd = best_rot
                    
                    placed.append((bx, by, bz, bw, bh, bd))
                    results[idx] = (0, bx, by, bz, bw, bh, bd)
                    
                    # Generate new candidate points
                    candidates.add((bx, by, bz + bd))
                    candidates.add((bx + bw, by, bz))
                    candidates.add((bx, by + bh, bz))
                    candidates.add((bx + bw, by + bh, bz))
                    candidates.add((bx + bw, by, bz + bd))
                    candidates.add((bx, by + bh, bz + bd))
                    candidates.add((bx + bw, by + bh, bz + bd))
            
            # Calculate score for this attempt
            total_filled = sum(r[1] * r[2] * r[3] for r in results if r[0] == 0)
            fraction = total_filled / container_volume if container_volume > 0 else 0
            
            if fraction > best_score_overall:
                best_score_overall = fraction
                best_result_overall = results
            
            attempt_count += 1
            
            # If we're using a lot of time per attempt, limit attempts
            elapsed = time.time() - start_time
            if elapsed > 30 and attempt_count > 10:
                break
        if time.time() > deadline:
            break
    
    return best_result_overall
