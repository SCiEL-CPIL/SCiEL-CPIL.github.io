import time
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Identify all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # To speed up, we can iterate and check
    # 128 choose 3 is ~349k. 
    # lcm calculation is fast.
    
    # We will store triples as tuples of indices or values?
    # Values are unique, so values are fine.
    
    # Optimization: 
    # For a fixed L, find all divisors of L in [1, 128].
    # If there are 3 divisors a,b,c of L such that lcm(a,b)=L, etc., they form a bad triple.
    # Actually, the condition is simply that they are divisors of L and for every prime power p^k || L, 
    # at least two of a,b,c are divisible by p^k.
    
    # Simpler: just iterate all triples.
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            lab = lcm(a, b)
            for k in range(j + 1, n):
                c = numbers[k]
                # Optimization: if lcm(a,c) != lab, continue
                # But we need to check all 3.
                # lcm(a,c) and lcm(b,c)
                # If lab is large, many c will fail.
                # Let's just compute.
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return list(range(1, n + 1)) # Fallback, though likely invalid, better to return prime powers
        
    # 2. Greedy removal
    # We want to remove minimum elements to break all bad triples.
    # This is Vertex Cover on Hypergraph.
    # Greedy: pick vertex in max number of uncovered edges.
    
    # Map value to index for fast access? Or just use values.
    # Let's use a set for current candidates.
    current_set = set(numbers)
    
    # We need to efficiently find bad triples in the current set.
    # Instead of re-scanning all triples, we can maintain a list of active bad triples.
    # But as we remove nodes, triples involving them become inactive.
    
    # Alternative: 
    # While there exists a bad triple in current_set:
    #   Find one.
    #   Remove the "most central" node from it.
    
    # To find a bad triple quickly, we can keep a list of all bad triples and check if all 3 are in current_set.
    # This is O(M) per step, where M is number of bad triples.
    # M might be large.
    
    # Let's estimate M.
    # For L=60, divisors are 1,2,3,4,5,6,10,12,15,20,30,60.
    # Bad triples for L=60?
    # {2,3,6}, {2,5,10}, {3,5,15}, {4,3,12}, {4,5,20}, {3,4,12}...
    # There are many.
    
    # Let's try the greedy removal on the list of bad triples.
    
    # Sort bad triples?
    # We will iterate through bad_triples, and if all 3 are in current_set, we have a conflict.
    
    # To be efficient, we can count "active" conflicts.
    # But simple check is fine if we limit iterations.
    
    # Let's use a priority queue or just pick the first one found.
    # To improve, pick the node with highest degree in the subgraph of active conflicts.
    
    # 1. Build adjacency list of conflicts?
    # Node u conflicts with v if there is a w such that (u,v,w) is bad.
    # This is the 2-shadow.
    
    # Let's stick to the triple list.
    
    # To speed up "find a bad triple in current_set":
    # We can maintain a counter for each triple? No, just check.
    
    # Optimization: 
    # If we remove a node, we invalidate all triples containing it.
    
    # Let's try a different approach:
    # Start with all nodes.
    # Calculate degree of each node (number of bad triples it belongs to).
    # While there are bad triples:
    #   Find a bad triple (a,b,c) such that a,b,c are in current_set.
    #   Among a,b,c, pick the one with highest degree.
    #   Remove it.
    #   Update degrees of all nodes in triples with it.
    
    # This requires knowing which triples involve a node.
    
    # Precompute: node_to_triples = {val: [indices of triples containing val]}
    
    node_to_triples = {x: [] for x in numbers}
    for idx, (a, b, c) in enumerate(bad_triples):
        node_to_triples[a].append(idx)
        node_to_triples[b].append(idx)
        node_to_triples[c].append(idx)
        
    # Current degree = number of active triples involving node.
    # Active triple = all 3 nodes in current_set.
    
    degrees = {x: len(node_to_triples[x]) for x in numbers}
    
    # We need to find *an* active triple.
    # We can iterate through all triples and check if all 3 are in current_set.
    # This is O(M) per removal.
    # M can be up to ~10k-50k?
    # 128^3 / 6 is 350k.
    # If we remove 50 nodes, 50 * 350k = 17.5M checks.
    # Each check is 3 set lookups. Fast in Python? Maybe.
    # 17.5M operations might take 1-2 seconds. Acceptable.
    
    # Let's refine:
    # Instead of scanning all triples, we can pick a random node from current_set, 
    # look at its triples, and see if any are active.
    
    # Or, simpler:
    # Just scan the list of bad_triples.
    
    while True:
        if time.time() > deadline:
            break
            
        found_triple = None
        # Find first active triple
        for t in bad_triples:
            if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                found_triple = t
                break
                
        if found_triple is None:
            break
            
        a, b, c = found_triple
        # Pick node with max degree
        max_deg_node = max([a, b, c], key=lambda x: degrees[x])
        
        current_set.remove(max_deg_node)
        
        # Update degrees for nodes in triples containing max_deg_node
        for t_idx in node_to_triples[max_deg_node]:
            ta, tb, tc = bad_triples[t_idx]
            # Decrease degree for ta, tb, tc if they are still in current_set?
            # Actually, degree is just a heuristic. 
            # The "active degree" is the number of active triples.
            # Removing max_deg_node makes all triples containing it inactive.
            # So for each neighbor (other nodes in those triples), their active degree decreases by 1 
            # (for each triple that was active).
            
            # But we don't track which triples are active efficiently.
            # Let's just recalculate degrees? No, too slow.
            
            # Let's just use the static degree as a proxy, or update it roughly.
            # If we remove a node, the triples it was in are gone.
            # So for any node u in a triple with max_deg_node, 
            # if that triple was active (i.e., u and the 3rd node are in current_set), 
            # then u's active degree drops.
            
            # To keep it simple and fast, we will just re-evaluate the "max degree" choice 
            # using the static degrees, or just pick randomly from the triple if degrees are similar.
            # Static degrees are a good enough heuristic for greedy.
            
            # However, to improve, we can decrement degrees of neighbors.
            # We need to know if the triple was active BEFORE removal.
            # It was active because we just found it (for the specific one) 
            # but for other triples containing max_deg_node, they might have been active or not.
            
            # Let's just decrement degrees of all neighbors in node_to_triples[max_deg_node] 
            # that are still in current_set.
            
            for neighbor in [ta, tb, tc]:
                if neighbor == max_deg_node:
                    continue
                if neighbor in current_set:
                    degrees[neighbor] -= 1
                    
    return list(current_set)
