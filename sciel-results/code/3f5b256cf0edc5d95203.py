import numpy as np
import random

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        # Autoconvolution
        c = np.convolve(h, h, mode='full')
        # Supremum norm of c
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    max_k = 25 # Max number of atoms
    
    for k_atoms in range(1, max_k + 1):
        best_local_score = 0.0
        best_local_h = None
        
        for trial in range(100): # Reduced trials to ensure speed
            # Generate random partition into k_atoms blocks
            if k_atoms - 1 >= n:
                continue
            cuts = sorted(random.sample(range(1, n), k_atoms - 1))
            starts = [0] + cuts
            ends = cuts + [n - 1]
            
            # Ensure valid widths
            valid = True
            for i in range(k_atoms):
                if ends[i] < starts[i]:
                    valid = False
                    break
            if not valid:
                continue
                
            h = np.zeros(n)
            total_width = sum(e - s + 1 for s, e in zip(starts, ends))
            if total_width == 0:
                continue
            for i in range(k_atoms):
                s, e = starts[i], ends[i]
                h[s:e+1] = 1.0 / total_width
            
            current_score = calc_score(h)
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_h = h.copy()
                best_local_starts = starts.copy()
                best_local_ends = ends.copy()
            
            # Hill climbing on boundaries
            improved = True
            max_iter = 50
            iter_count = 0
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                for i in range(k_atoms):
                    # Move start
                    old_s = starts[i]
                    for delta in [-1, 1]:
                        new_s = old_s + delta
                        if new_s < 0 or new_s > ends[i]:
                            continue
                        if i > 0 and new_s <= ends[i-1]:
                            continue
                        if i < k_atoms - 1 and new_s >= starts[i+1]:
                            continue
                        
                        old_starts = starts.copy()
                        starts[i] = new_s
                        
                        total_width = sum(e - s + 1 for s, e in zip(starts, ends))
                        if total_width == 0:
                            starts = old_starts
                            continue
                        h_new = np.zeros(n)
                        for j in range(k_atoms):
                            s_j, e_j = starts[j], ends[j]
                            if s_j <= e_j:
                                h_new[s_j:e_j+1] = 1.0 / total_width
                        score_new = calc_score(h_new)
                        
                        if score_new > current_score:
                            current_score = score_new
                            starts = starts.copy()
                            h = h_new
                            improved = True
                            if current_score > best_local_score:
                                best_local_score = current_score
                                best_local_h = h.copy()
                                best_local_starts = starts.copy()
                                best_local_ends = ends.copy()
                        else:
                            starts = old_starts

                    # Move end
                    old_e = ends[i]
                    for delta in [-1, 1]:
                        new_e = old_e + delta
                        if new_e >= n or new_e < starts[i]:
                            continue
                        if i < k_atoms - 1 and new_e >= starts[i+1]:
                            continue
                        # Note: if i > 0, we don't constrain new_e by starts[i-1] directly, 
                        # but by the fact that starts[i] > ends[i-1] is already maintained 
                        # by the start move constraints and initial valid state.
                        # Wait, if we move end[i] left, it might violate end[i] >= start[i]. 
                        # Checked above.
                        # Does it violate end[i] < start[i+1]? 
                        # If we move left, it gets smaller, so if it was < start[i+1], it remains <.
                        # If we move right, we check new_e < start[i+1].
                        
                        old_ends = ends.copy()
                        ends[i] = new_e
                        
                        total_width = sum(e - s + 1 for s, e in zip(starts, ends))
                        if total_width == 0:
                            ends = old_ends
                            continue
                        h_new = np.zeros(n)
                        for j in range(k_atoms):
                            s_j, e_j = starts[j], ends[j]
                            if s_j <= e_j:
                                h_new[s_j:e_j+1] = 1.0 / total_width
                        score_new = calc_score(h_new)
                        
                        if score_new > current_score:
                            current_score = score_new
                            ends = ends.copy()
                            h = h_new
                            improved = True
                            if current_score > best_local_score:
                                best_local_score = current_score
                                best_local_h = h.copy()
                                best_local_starts = starts.copy()
                                best_local_ends = ends.copy()
                        else:
                            ends = old_ends

        if best_local_h is not None:
            if best_local_score > best_score:
                best_score = best_local_score
                best_h = best_local_h.copy()
                
    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)
