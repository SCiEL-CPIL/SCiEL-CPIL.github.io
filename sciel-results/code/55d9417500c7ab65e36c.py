import time
import sys
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict graph on edges
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)

        # conflict[i] = set of j such that edge i and edge j conflict
        # (share a vertex or are joined by an edge)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                if a == c or a == dd or b == c or b == dd:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        # Initial coloring: greedy with highest-conflict-degree first
        deg_conf = [len(conflict[i]) for i in range(m)]
        order = sorted(range(m), key=lambda x: deg_conf[x], reverse=True)

        def greedy_color(order):
            color = [-1] * m
            for v in order:
                forbidden = set()
                for u in conflict[v]:
                    if color[u] != -1:
                        forbidden.add(color[u])
                c = 0
                while c in forbidden:
                    c += 1
                color[v] = c
            return color

        color = greedy_color(order)
        K = max(color) + 1

        # Local search: simulated annealing to reduce K by moving vertices
        # Cost = number of conflicts (should be 0 for valid coloring)
        # We want to minimize K while keeping 0 conflicts.

        def count_conflicts(color):
            cnt = 0
            for i in range(m):
                ci = color[i]
                for j in conflict[i]:
                    if j > i and color[j] == ci:
                        cnt += 1
            return cnt

        # Verify initial coloring is valid
        assert count_conflicts(color) == 0, "Initial coloring has conflicts"

        # Simulated annealing: try to merge color classes
        # State: color array, K = max+1
        # Move: pick a vertex, change its color to some other color in [0, K-1]
        # We want to reduce K: if a color class becomes empty, K decreases.

        rng = Random(seed * 104729 + 7)
        best_color = color[:]
        best_K = K

        # Phase 1: Annealing to minimize K
        # Start with high temperature, cool down
        # Energy = K (we want to minimize this)
        # But we must maintain 0 conflicts.
        
        # Alternative approach: directly try to find K-coloring by
        # starting from current coloring and trying to eliminate one color at a time.

        # For each target K from current_K-1 down to lower bound:
        # Try to recolor to use at most target_K colors.
        
        lower_bound = max(deg_conf) + 1  # chromatic number lower bound

        current_color = color[:]
        current_K = K

        def try_reduce_to(target_K, max_time=10.0):
            """Try to find a valid coloring with at most target_K colors."""
            if time.time() > deadline - 2.0:
                return None
            
            # Start from current coloring, remap colors to [0, target_K-1]
            # First, compress current colors
            c = current_color[:]
            
            # Remap: if any color >= target_K, we need to reassign those vertices
            # Strategy: simulated annealing where we can only use colors 0..target_K-1
            # and we pay a penalty for conflicts.
            
            # Initial: assign each vertex a random color in [0, target_K-1]
            # but start from current assignment where possible
            c = c[:]
            for i in range(m):
                if c[i] >= target_K:
                    c[i] = rng.randrange(target_K)
            
            # Compute initial conflict count
            def get_conflict_count(c):
                cnt = 0
                for i in range(m):
                    ci = c[i]
                    for j in conflict[i]:
                        if j > i and c[j] == ci:
                            cnt += 1
                return cnt
            
            cur_conf = get_conflict_count(c)
            cur_K_used = len(set(c))
            
            # If already valid with <= target_K colors, done
            if cur_conf == 0:
                return c
            
            # Simulated annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.001
            steps = 0
            max_steps = 500000
            best_c = c[:]
            best_conf = cur_conf
            
            # Track if we ever reach 0 conflicts
            found_valid = False
            
            while T > min_T and steps < max_steps:
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random vertex
                v = rng.randrange(m)
                old_color = c[v]
                
                # Pick a new color (different from old)
                if target_K > 1:
                    new_color = rng.randrange(target_K)
                    if new_color == old_color:
                        new_color = (new_color + 1) % target_K
                else:
                    new_color = 0
                
                # Compute delta in conflicts
                # Conflicts involving v with color old_color decrease
                # Conflicts involving v with color new_color increase
                delta = 0
                for u in conflict[v]:
                    if c[u] == old_color:
                        delta -= 1  # removing a conflict
                    if c[u] == new_color:
                        delta += 1  # adding a conflict
                
                if delta <= 0 or rng.random() < __import__('math').exp(-delta / T):
                    c[v] = new_color
                    cur_conf += delta
                    if cur_conf < best_conf:
                        best_conf = cur_conf
                        best_c = c[:]
                    if cur_conf == 0:
                        found_valid = True
                        # Keep going to try to reduce further, but we have a valid solution
                        # Actually, if we have 0 conflicts, we're done for this target_K
                        break
                
                T *= cooling
                steps += 1
            
            if found_valid and get_conflict_count(c) == 0:
                return c
            if best_conf == 0:
                return best_c
            return None

        # Try to reduce K step by step
        for target_K in range(current_K - 1, lower_bound - 1, -1):
            if time.time() > deadline - 3.0:
                break
            
            result = try_reduce_to(target_K)
            if result is not None:
                current_color = result
                current_K = target_K
                # Continue trying to reduce further
            else:
                # Try a few more times with different random seeds
                success = False
                for attempt in range(5):
                    if time.time() > deadline - 3.0:
                        break
                    result = try_reduce_to(target_K)
                    if result is not None:
                        current_color = result
                        current_K = target_K
                        success = True
                        break
                if not success:
                    break  # Can't reduce further
        
        return compress(current_color)

    try:
        start = time.time()
        deadline = start + 140.0  # Safety margin before 150s kill
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            if remaining < 5.0:
                # Not enough time, use fallback for remaining
                for s in specs[idx:]:
                    edges = instance(*s)
                    adj = {}
                    for u, v in edges:
                        adj.setdefault(u, set()).add(v)
                        adj.setdefault(v, set()).add(u)
                    colour = {}
                    for e in edges:
                        near = {x for x in e} | {y for x in e for y in adj[x]}
                        taken = {c for f, c in colour.items() if set(f) & near}
                        colour[e] = next(c for c in range(len(edges)) if c not in taken)
                    out.append([colour[e] for e in edges])
                return out
            share = remaining / (len(specs) - idx)
            share = min(share, 45.0)
            share = max(share, 5.0)
            instance_deadline = time.time() + share
            out.append(solve_instance(n, d, seed, instance_deadline))
        return out
    except Exception:
        return fallback()
