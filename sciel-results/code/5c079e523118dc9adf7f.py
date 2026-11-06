import time
import numpy as np
import random
from collections import Counter

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4s.append((e1, e2, e3, e4, e5, e6))
    
    num_k4s = len(k4s)
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(k4_idx)
    
    # Precompute for each K4, the position of each edge in the K4
    k4_edge_pos = [{} for _ in range(num_k4s)]
    for k4_idx, k4 in enumerate(k4s):
        for pos, e in enumerate(k4):
            k4_edge_pos[k4_idx][e] = pos
    
    deadline = time.time() + 115
    
    # Start from the known valid coloring
    colors = [(i + j) % 17 for (i, j) in edges]
    
    # Track distinct color counts per K4 using Counter
    k4_counters = [Counter() for _ in range(num_k4s)]
    for k4_idx, k4 in enumerate(k4s):
        for e in k4:
            k4_counters[k4_idx][colors[e]] += 1
    
    def get_violations():
        viol = []
        for k4_idx in range(num_k4s):
            if len(k4_counters[k4_idx]) < 5:
                viol.append(k4_idx)
        return viol
    
    def palette_size():
        return len(set(colors))
    
    best_colors = colors[:]
    best_pal = palette_size()
    
    # Phase 1: Try to reduce palette from 17 down
    # Strategy: pick a color, try to remap all its edges to other colors
    
    rng = random.Random(42)
    
    # Try multiple target palette sizes
    for target_K in [15, 14, 13]:
        if time.time() > deadline:
            break
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Start from best known or random perturbation
            if restart == 0:
                cur = best_colors[:]
            else:
                cur = best_colors[:]
                # Random perturbation
                for _ in range(rng.randint(1, 20)):
                    e = rng.randrange(num_edges)
                    cur[e] = rng.randint(0, target_K - 1)
            
            # Rebuild counters
            k4_counters = [Counter() for _ in range(num_k4s)]
            for k4_idx, k4 in enumerate(k4s):
                for e in k4:
                    k4_counters[k4_idx][cur[e]] += 1
            
            viol = get_violations()
            
            if not viol and palette_size() <= target_K:
                if palette_size() < best_pal:
                    best_colors = cur[:]
                    best_pal = palette_size()
                break
            
            # Local search: reduce violations, then reduce palette
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                pal = set(cur)
                pal_size = len(pal)
                
                if not viol and pal_size <= target_K:
                    if pal_size < best_pal:
                        best_colors = cur[:]
                        best_pal = pal_size
                    break
                
                if viol:
                    # Pick a violated K4
                    k4_idx = rng.choice(viol)
                    k4 = k4s[k4_idx]
                    
                    best_delta = 0
                    best_e = -1
                    best_c = -1
                    
                    for e in k4:
                        old_c = cur[e]
                        # Try colors: existing palette + a few new ones
                        candidates = list(pal)
                        if len(candidates) < target_K:
                            candidates.extend(range(len(candidates), target_K))
                        
                        for nc in candidates:
                            if nc == old_c:
                                continue
                            
                            # Compute delta: how does distinct count change?
                            cnt = k4_counters[k4_idx]
                            old_count = cnt[old_c]
                            new_count = cnt.get(nc, 0)
                            
                            # After change:
                            # old_c count decreases by 1
                            # nc count increases by 1
                            
                            # Distinct before
                            old_distinct = len(cnt)
                            
                            # Distinct after
                            new_distinct = old_distinct
                            if old_count == 1:
                                new_distinct -= 1  # old_c disappears
                            if new_count == 0:
                                new_distinct += 1  # nc is new
                            
                            # Delta in violation status
                            old_valid = (old_distinct >= 5)
                            new_valid = (new_distinct >= 5)
                            
                            if old_valid and not new_valid:
                                delta = 1
                            elif not old_valid and new_valid:
                                delta = -1
                            else:
                                delta = 0
                            
                            # Also consider palette size impact
                            if not viol and pal_size > target_K:
                                # Prefer moves that reduce palette
                                if nc not in pal and old_count == 1:
                                    delta -= 0.5  # slight preference
                            
                            if delta < best_delta:
                                best_delta = delta
                                best_e = e
                                best_c = nc
                    
                    if best_e != -1:
                        old_c = cur[best_e]
                        cur[best_e] = best_c
                        
                        # Update counters
                        for k4_idx_affected in edge_to_k4s[best_e]:
                            cnt = k4_counters[k4_idx_affected]
                            cnt[old_c] -= 1
                            if cnt[old_c] == 0:
                                del cnt[old_c]
                            cnt[best_c] = cnt.get(best_c, 0) + 1
                        
                        # Rebuild violation set
                        viol = get_violations()
                    else:
                        # Random move
                        e = rng.choice(k4)
                        nc = rng.randint(0, target_K - 1)
                        if nc != cur[e]:
                            old_c = cur[e]
                            cur[e] = nc
                            for k4_idx_affected in edge_to_k4s[e]:
                                cnt = k4_counters[k4_idx_affected]
                                cnt[old_c] -= 1
                                if cnt[old_c] == 0:
                                    del cnt[old_c]
                                cnt[nc] = cnt.get(nc, 0) + 1
                            viol = get_violations()
                else:
                    # No violations, try to reduce palette
                    if pal_size <= target_K:
                        break
                    
                    # Pick a color to try to eliminate
                    pal_list = list(pal)
                    target_color = rng.choice(pal_list)
                    edges_with_target = [e for e, c in enumerate(cur) if c == target_color]
                    
                    if not edges_with_target:
                        continue
                    
                    # Try to remap each edge
                    success = True
                    for e in edges_with_target:
                        found = False
                        other_colors = [c for c in pal_list if c != target_color]
                        rng.shuffle(other_colors)
                        # Also try new colors if needed
                        if len(other_colors) < 3:
                            other_colors.extend(range(max(pal_list) + 1, target_K))
                        
                        for nc in other_colors:
                            # Check if this move keeps all K4s valid
                            valid = True
                            for k4_idx_affected in edge_to_k4s[e]:
                                cnt = k4_counters[k4_idx_affected]
                                old_c = cur[e]
                                old_count = cnt[old_c]
                                new_count = cnt.get(nc, 0)
                                old_distinct = len(cnt)
                                new_distinct = old_distinct
                                if old_count == 1:
                                    new_distinct -= 1
                                if new_count == 0:
                                    new_distinct += 1
                                if new_distinct < 5:
                                    valid = False
                                    break
                            if valid:
                                found = True
                                # Apply
                                old_c = cur[e]
                                cur[e] = nc
                                for k4_idx_affected in edge_to_k4s[e]:
                                    cnt = k4_counters[k4_idx_affected]
                                    cnt[old_c] -= 1
                                    if cnt[old_c] == 0:
                                        del cnt[old_c]
                                    cnt[nc] = cnt.get(nc, 0) + 1
                                break
                        
                        if not found:
                            success = False
                            break
                    
                    if success:
                        # Verify
                        viol = get_violations()
                        if not viol:
                            new_pal = len(set(cur))
                            if new_pal < best_pal:
                                best_colors = cur[:]
                                best_pal = new_pal
                        break
                    else:
                        # Failed, try a different approach: random edge change
                        e = rng.randrange(num_edges)
                        nc = rng.randint(0, target_K - 1)
                        if nc != cur[e]:
                            old_c = cur[e]
                            cur[e] = nc
                            for k4_idx_affected in edge_to_k4s[e]:
                                cnt = k4_counters[k4_idx_affected]
                                cnt[old_c] -= 1
                                if cnt[old_c] == 0:
                                    del cnt[old_c]
                                cnt[nc] = cnt.get(nc, 0) + 1
        
        # Check if we achieved target
        if best_pal <= target_K:
            break
    
    # Final validation and return
    # Ensure colors are in 0..119
    max_c = max(best_colors)
    if max_c > 119:
        # Remap
        color_map = {c: i for i, c in enumerate(sorted(set(best_colors)))}
        best_colors = [color_map[c] for c in best_colors]
    
    return best_colors
