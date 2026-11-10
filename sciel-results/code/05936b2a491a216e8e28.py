import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Vertices 0..14
    # Base graph: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
            
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid(edges):
        deg = get_degrees(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all paths s-u-v-t
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Initial best solution
    best_edges = set(base_edges)
    best_count = len(best_edges)
    
    # All possible non-base edges (internal to A or B)
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append(tuple(sorted((A[i], A[j]))))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append(tuple(sorted((B[i], B[j]))))
    
    # Local search: try removing one edge and adding up to k others
    def local_search(edges, max_adds=2, time_limit=None):
        improved = True
        while improved:
            if time_limit and time.time() > time_limit:
                break
            improved = False
            edge_list = list(edges)
            random.shuffle(edge_list)
            for rem in edge_list:
                if time_limit and time.time() > time_limit:
                    break
                without_rem = edges - {rem}
                # Try adding 0, 1, or 2 edges
                for add_k in range(max_adds + 1):
                    if time_limit and time.time() > time_limit:
                        break
                    if add_k == 0:
                        # Just remove
                        if len(without_rem) > len(edges):
                            # Can't happen since we're removing
                            pass
                        # Only useful if removing allows more additions later
                        # For now, skip pure removals unless they enable better adds
                        continue
                    # Try adding 'add_k' edges from candidates not in without_rem
                    avail = [c for c in candidates if c not in without_rem and c != rem]
                    random.shuffle(avail)
                    # Try subsets of size add_k from first few candidates
                    if add_k == 1:
                        for add_e in avail[:30]:
                            if time_limit and time.time() > time_limit:
                                break
                            test = without_rem | {add_e}
                            if len(test) > len(edges) and is_valid(test):
                                edges = test
                                improved = True
                                break
                        if improved:
                            break
                    elif add_k == 2:
                        # Try pairs from a small subset
                        small_avail = avail[:15]
                        for i in range(len(small_avail)):
                            if time_limit and time.time() > time_limit:
                                break
                            for j in range(i+1, len(small_avail)):
                                if time_limit and time.time() > time_limit:
                                    break
                                test = without_rem | {small_avail[i], small_avail[j]}
                                if len(test) > len(edges) and is_valid(test):
                                    edges = test
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        return edges

    # Greedy add pass
    def greedy_add(edges, candidates_order):
        for c in candidates_order:
            if c in edges:
                continue
            test = edges | {c}
            if is_valid(test):
                edges = test
        return edges

    # Initial greedy
    current_edges = set(base_edges)
    shuffled_cands = candidates[:]
    random.shuffle(shuffled_cands)
    current_edges = greedy_add(current_edges, shuffled_cands)
    
    # Local search on initial result
    current_edges = local_search(current_edges, max_adds=2, time_limit=start_time + 30.0)
    
    if len(current_edges) > best_count:
        best_edges = set(current_edges)
        best_count = len(best_edges)
    
    # Random restarts with local search
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Start from base or from best with some randomization
        if restart % 3 == 0:
            curr = set(base_edges)
        elif restart % 3 == 1:
            # Start from best, remove 1-3 random edges
            curr = set(best_edges)
            to_remove = random.sample(list(curr), min(random.randint(1,3), len(curr)))
            for e in to_remove:
                curr.discard(e)
        else:
            # Start from base, add a few random edges first
            curr = set(base_edges)
            random_cands = candidates[:]
            random.shuffle(random_cands)
            for c in random_cands[:random.randint(0, 5)]:
                test = curr | {c}
                if is_valid(test):
                    curr = test
        
        # Greedy add
        shuffled = candidates[:]
        random.shuffle(shuffled)
        curr = greedy_add(curr, shuffled)
        
        # Local search
        remaining_time = deadline - time.time()
        if remaining_time > 2:
            curr = local_search(curr, max_adds=2, time_limit=start_time + deadline - 2.0)
        
        if len(curr) > best_count:
            best_edges = set(curr)
            best_count = len(curr)
    
    result = [list(e) for e in best_edges]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
        
    return result
