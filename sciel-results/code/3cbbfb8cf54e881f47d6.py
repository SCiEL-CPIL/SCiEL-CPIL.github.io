import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 295  # 5s safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(all_points)
    
    best_set = []
    best_size = 0
    
    def can_insert(px, py, selected, dist_from):
        """
        Check if point (px, py) can be added to the selected set.
        Returns (valid, d_list) where d_list[i] is squared dist from (px,py) to selected[i].
        """
        d_list = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_list.append(d_sq)
        
        # Check 1: P is apex. All distances from P must be unique.
        if len(set(d_list)) != len(d_list):
            return False, d_list
        
        # Check 2: For each selected point A, d(P,A) must not equal d(A,B) for any other B.
        for i in range(len(selected)):
            d_pa = d_list[i]
            if d_pa in dist_from[i]:
                return False, d_list
        
        return True, d_list

    def insert_point(p, selected, dist_from):
        """Insert point p into selected, updating dist_from. Assumes can_insert returned True."""
        px, py = p
        d_list = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_list.append(d_sq)
        
        selected.append(p)
        dist_from.append(set(d_list))
        for i in range(len(selected) - 1):
            dist_from[i].add(d_list[i])
    
    def remove_point(idx, selected, dist_from):
        """Remove point at index idx from selected, updating dist_from."""
        selected.pop(idx)
        dist_from.pop(idx)
        # Rebuild dist_from for remaining points (simpler than tracking removals)
        new_dist_from = []
        for i in range(len(selected)):
            s = set()
            for j in range(len(selected)):
                if i == j:
                    continue
                dx = selected[i][0] - selected[j][0]
                dy = selected[i][1] - selected[j][1]
                s.add(dx*dx + dy*dy)
            new_dist_from.append(s)
        dist_from.clear()
        dist_from.extend(new_dist_from)
    
    def greedy_with_order(order, time_limit=None):
        """
        Greedy selection following the given order.
        Returns (selected, dist_from).
        """
        selected = []
        dist_from = []
        
        for p in order:
            if time_limit is not None and time.time() > time_limit:
                break
            valid, d_list = can_insert(p[0], p[1], selected, dist_from)
            if valid:
                insert_point(p, selected, dist_from)
        
        return selected, dist_from

    def backtrack_search(selected, dist_from, candidates, depth=0, time_limit=None):
        """
        Backtracking search to find a larger valid set.
        candidates is a list of points not yet in selected.
        Returns (selected, dist_from, improved) where improved is True if we found a larger set.
        """
        if time_limit is not None and time.time() > time_limit:
            return selected, dist_from, False
        
        best_local = list(selected)
        best_local_dist = [set(s) for s in dist_from]
        best_size_local = len(selected)
        
        for ci in range(len(candidates)):
            if time_limit is not None and time.time() > time_limit:
                break
            
            p = candidates[ci]
            valid, d_list = can_insert(p[0], p[1], selected, dist_from)
            if not valid:
                continue
            
            # Insert
            insert_point(p, selected, dist_from)
            
            # Recurse with remaining candidates
            remaining = candidates[:ci] + candidates[ci+1:]
            new_selected, new_dist, improved = backtrack_search(
                selected, dist_from, remaining, depth+1, time_limit
            )
            
            if len(new_selected) > best_size_local:
                best_size_local = len(new_selected)
                best_local = new_selected
                best_local_dist = [set(s) for s in new_dist]
            
            # Backtrack: remove p
            remove_point(len(selected)-1, selected, dist_from)
            
            # If we found a larger set, update and continue searching from there
            if len(best_local) > len(selected):
                selected.clear()
                selected.extend(best_local)
                dist_from.clear()
                dist_from.extend(best_local_dist)
                # Recompute candidates
                sel_set = set(selected)
                candidates = [c for c in candidates if c not in sel_set]
        
        return selected, dist_from, len(selected) > depth

    def get_diversity_order(selected, all_pts):
        """
        Order candidates by maximum minimum distance to selected points (diversity).
        Points far from the existing set are tried first.
        """
        if not selected:
            return list(all_pts)
        
        scored = []
        for p in all_pts:
            if p in selected:
                continue
            min_dist = float('inf')
            for s in selected:
                dx = p[0] - s[0]
                dy = p[1] - s[1]
                d = dx*dx + dy*dy
                if d < min_dist:
                    min_dist = d
            scored.append((min_dist, p))
        
        # Sort by decreasing min distance (most diverse first)
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    # Strategy: Multiple restarts with greedy initialization + backtracking improvement
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        remaining_time = deadline - 10 - time.time()
        
        # Random shuffle for diversity
        shuffled = list(all_points)
        random.shuffle(shuffled)
        
        # Greedy initialization
        selected, dist_from = greedy_with_order(shuffled, time_limit=deadline - 15)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
        
        # Backtracking improvement if we have time
        if time.time() < deadline - 15 and len(selected) < 40:
            # Get diverse candidate ordering
            sel_set = set(selected)
            candidates = [p for p in shuffled if p not in sel_set]
            
            # Limit candidates to top diverse ones for efficiency
            if len(candidates) > 500:
                candidates = get_diversity_order(selected, candidates)[:500]
            
            bt_time_limit = deadline - 10
            new_selected, new_dist, improved = backtrack_search(
                selected, dist_from, candidates, 0, bt_time_limit
            )
            
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = list(new_selected)
        
        # Occasionally try a fresh greedy with different ordering
        if time.time() < deadline - 15 and restart_count % 3 == 0:
            # Sort by some other heuristic: e.g., by x+y mod something
            alt_order = sorted(all_points, key=lambda p: (p[0] * 7 + p[1] * 13) % 97)
            random.shuffle(alt_order)
            sel2, _ = greedy_with_order(alt_order, time_limit=deadline - 15)
            if len(sel2) > best_size:
                best_size = len(sel2)
                best_set = list(sel2)
        
        if time.time() >= deadline - 10:
            break
    
    return best_set
