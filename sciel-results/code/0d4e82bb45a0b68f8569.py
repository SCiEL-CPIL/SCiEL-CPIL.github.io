"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """Return three lists of paths (vertex lists) for the three fixed instance graphs."""
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """Decompose edges into edge-disjoint simple paths covering all edges."""
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))

        # Build initial paths: each edge is a path of length 1
        # We'll work with path fragments and merge them.
        
        # Represent each path as a list of vertices
        # Start with each edge as its own path
        paths = [[u, v] for u, v in edges]
        
        # Track which edges are used (all are used initially)
        # We need to merge paths at endpoints.
        
        # For merging: two paths can be merged if they share an endpoint
        # and the merge doesn't create a cycle (i.e., the other endpoints are different
        # and the shared vertex is an endpoint of both paths).
        
        # Actually, let's think more carefully.
        # A path is a sequence of vertices where consecutive vertices are connected by edges.
        # Two paths P1 = [a, ..., b] and P2 = [c, ..., d] can be merged if:
        #   - b == c or b == d or a == c or a == d (they share an endpoint)
        #   - The merge doesn't create a cycle: if we merge at b==c, the new path is [a,...,b,...,d]
        #     This is valid as long as a != d (otherwise it would be a cycle, not a path)
        #   - All edges in the merged path are distinct (they are, since we're merging edge-disjoint paths)
        #   - The merged path is simple (no repeated vertices)
        
        # Let's implement a greedy merging strategy:
        # Repeatedly try to merge pairs of paths that share an endpoint.
        
        def can_merge(p1, p2):
            """Check if two paths can be merged into a simple path."""
            # Find shared endpoints
            # p1 = [a, ..., b], p2 = [c, ..., d]
            # Possible merges:
            # 1. b == c: merge as p1 + p2[1:] -> [a,...,b,c,...,d] but c==b so [a,...,b,...,d]
            #    Wait, p2[1:] starts with the vertex after c. If p2 = [c, x, ..., d], then p2[1:] = [x, ..., d]
            #    Merged: [a, ..., b, x, ..., d] where b==c. Need to check simplicity.
            # 2. b == d: merge as p1 + reversed(p2[1:]) -> [a,...,b,d,...,c] where d==b
            # 3. a == c: merge as p1 + p2[1:] -> [c,...,b, x,...,d] wait, this is reversed
            #    Actually if a==c, we can do reversed(p1) + p2[1:] or p1 + reversed(p2)... 
            #    Let me think again.
            
            # Let's define merge options:
            # Option 1: p1 ends at vertex that p2 starts at. p1[-1] == p2[0]
            #   New path: p1 + p2[1:]
            # Option 2: p1 ends at vertex that p2 ends at. p1[-1] == p2[-1]
            #   New path: p1 + reversed(p2[:-1])
            # Option 3: p1 starts at vertex that p2 starts at. p1[0] == p2[0]
            #   New path: p2 + reversed(p1[1:])  or  reversed(p1) + p2[1:]... 
            #   Actually: reversed(p1) + p2[1:] gives [b,...,a,...,d] where a==c, so [b,...,c,...,d]
            #   Or: p2 + reversed(p1[1:]) gives [c,...,d, b,...,a] where c==a, so [a,...,d, b,...,a] - no, that repeats a.
            #   Let me be more careful.
            
            # Let's just try all 4 combinations and check simplicity:
            # Combo 1: p1 + p2[1:]  (valid if p1[-1] == p2[0])
            # Combo 2: p1 + reversed(p2[:-1])  (valid if p1[-1] == p2[-1])
            # Combo 3: p2 + reversed(p1[1:])  (valid if p2[-1] == p1[0])  -- wait, this should be p2 + reversed(p1[1:]) only if p2[-1]==p1[0]
            #   Actually reversed(p1[1:]) = [b,...,p1[1]] and we prepend p2: [c,...,d, b,...,p1[1]]
            #   This works if d == p1[0], i.e., p2[-1] == p1[0].
            # Combo 4: p2 + reversed(p1[:-1])  (valid if p2[-1] == p1[-1])... no wait.
            
            # Let me redo this:
            # To merge p1 and p2 sharing a vertex v:
            # - If v = p1[-1] = p2[0]: result = p1 + p2[1:]
            # - If v = p1[-1] = p2[-1]: result = p1 + reversed(p2[:-1])
            # - If v = p1[0] = p2[0]: result = reversed(p1) + p2[1:]  -- no, that's [p1[-1],...,p1[0],p2[1],...]
            #   Hmm, reversed(p1) = [p1[-1],...,p1[0]] and p2[1:] = [p2[1],...,p2[-1]]
            #   So result = [p1[-1],...,v,p2[1],...,p2[-1]] where v=p1[0]=p2[0]. This is valid.
            # - If v = p1[0] = p2[-1]: result = reversed(p1) + p2[:-1]... 
            #   reversed(p1) = [p1[-1],...,p1[0]] and p2[:-1] = [p2[0],...,p2[-2]]
            #   Result = [p1[-1],...,v,p2[0],...,p2[-2]] where v=p1[0]=p2[-1]. 
            #   Wait, p2[0] is the start of p2. So we're going from v into p2's interior. That's fine.
            
            # Actually, I think the cleanest way:
            # For each of the 4 endpoint-pair matches, construct the candidate merged path and check if it's simple.
            
            candidates = []
            
            # p1[-1] == p2[0]
            if p1[-1] == p2[0]:
                cand = p1 + p2[1:]
                if len(set(cand)) == len(cand):
                    candidates.append(cand)
            
            # p1[-1] == p2[-1]
            if p1[-1] == p2[-1]:
                cand = p1 + list(reversed(p2[:-1]))
                if len(set(cand)) == len(cand):
                    candidates.append(cand)
            
            # p1[0] == p2[0]
            if p1[0] == p2[0]:
                cand = list(reversed(p1)) + p2[1:]
                if len(set(cand)) == len(cand):
                    candidates.append(cand)
            
            # p1[0] == p2[-1]
            if p1[0] == p2[-1]:
                cand = list(reversed(p1)) + p2[:-1]
                if len(set(cand)) == len(cand):
                    candidates.append(cand)
            
            return candidates
        
        def do_merge(p1, p2):
            """Merge two paths, return merged path or None."""
            cands = can_merge(p1, p2)
            if cands:
                # Pick the longest (should all be same length actually)
                return cands[0]
            return None
        
        # Greedy merging: repeatedly find a pair that can be merged and merge them.
        # Use a priority: prefer merges that create longer paths.
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Try all pairs
            merged_any = False
            i = 0
            while i < len(paths) and time.time() < deadline - 5.0:
                j = i + 1
                while j < len(paths) and time.time() < deadline - 5.0:
                    result = do_merge(paths[i], paths[j])
                    if result is not None:
                        paths[i] = result
                        paths.pop(j)
                        merged_any = True
                        # Restart from i (don't increment i since we replaced it)
                        break
                    j += 1
                if merged_any:
                    merged_any = False
                    continue
                i += 1
            changed = merged_any
        
        # Convert to list of lists
        return [list(p) for p in paths]

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    result = []
    for spec in specs:
        edges = instance(*spec)
        paths = decompose(edges)
        result.append(paths)
    
    return result
# EVOLVE-BLOCK-END
