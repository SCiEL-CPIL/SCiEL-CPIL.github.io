import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    # We will focus on bipartite graphs K_{a,b}
    # Try to find a valid coloring for K_{10,10}
    
    best_edges = []
    best_count = 0
    
    # Function to check if a graph is valid and return list of bad 4-cycles
    def get_bad_cycles(edges_dict, adj):
        bad_cycles = []
        # For K_{a,b}, vertices are 0..a-1 and a..a+b-1
        # Let's assume we are working with a specific bipartition
        # For generality in the loop, we'll pass the specific sets
        
        # Iterate over all pairs of left vertices and right vertices
        # This is specific to the bipartite structure we maintain
        
        return bad_cycles

    # We'll implement a specific solver for K_{10,10}
    a, b = 10, 10
    left = list(range(a))
    right = list(range(a, a+b))
    
    # Initialize with all edges of K_{10,10}
    edges_dict = {}
    adj = {i: set() for i in range(n)}
    
    for u in left:
        for v in right:
            edges_dict[(u, v)] = random.randint(0, 19)
            adj[u].add(v)
            adj[v].add(u)
    
    current_edges = list(edges_dict.keys())
    
    def count_bad_cycles(edges_dict, adj, left, right):
        bad = 0
        # Iterate over all 4-cycles in K_{a,b}
        # A 4-cycle is determined by 2 left vertices l1, l2 and 2 right vertices r1, r2
        # Edges: (l1,r1), (l1,r2), (l2,r1), (l2,r2)
        # Cycle: l1-r1-l2-r2-l1
        # Colors: c(l1,r1), c(l2,r1), c(l2,r2), c(l1,r2)
        # Must be distinct.
        
        for l1 in left:
            for l2 in left:
                if l1 >= l2: continue
                for r1 in right:
                    for r2 in right:
                        if r1 >= r2: continue
                        
                        # Check if all 4 edges exist
                        if (l1,r1) not in edges_dict: continue
                        if (l2,r1) not in edges_dict: continue
                        if (l2,r2) not in edges_dict: continue
                        if (l1,r2) not in edges_dict: continue
                        
                        c1 = edges_dict[(l1,r1)]
                        c2 = edges_dict[(l2,r1)]
                        c3 = edges_dict[(l2,r2)]
                        c4 = edges_dict[(l1,r2)]
                        
                        if len({c1, c2, c3, c4}) != 4:
                            bad += 1
        return bad

    # Initial bad count
    bad_count = count_bad_cycles(edges_dict, adj, left, right)
    
    # Local Search
    # We want to minimize bad_count. If bad_count == 0, we have a valid graph.
    # If we can't get 0, we remove edges to reduce bad_count.
    
    # Strategy:
    # 1. If bad_count > 0, try to fix.
    # 2. With probability p, recolor an edge involved in a bad cycle.
    # 3. With probability 1-p, remove an edge involved in a bad cycle.
    
    # To pick an edge involved in a bad cycle, we need to find one.
    def find_bad_cycle(edges_dict, left, right):
        for l1 in left:
            for l2 in left:
                if l1 >= l2: continue
                for r1 in right:
                    for r2 in right:
                        if r1 >= r2: continue
                        if (l1,r1) in edges_dict and (l2,r1) in edges_dict and (l2,r2) in edges_dict and (l1,r2) in edges_dict:
                            c1 = edges_dict[(l1,r1)]
                            c2 = edges_dict[(l2,r1)]
                            c3 = edges_dict[(l2,r2)]
                            c4 = edges_dict[(l1,r2)]
                            if len({c1, c2, c3, c4}) != 4:
                                return [(l1,r1), (l2,r1), (l2,r2), (l1,r2)]
        return None

    # Optimization: Precompute lists for faster iteration?
    # For now, the O(N^4) check is ~ 10^4 * 4 = 40k ops per check.
    # 1000 iterations * 40k = 40M ops. Might be slow in Python if not careful.
    # Let's optimize by only checking if we change something.
    
    # Better: Maintain a list of all potential 4-cycles (tuples of 4 edges).
    # There are ~2000 such cycles.
    potential_cycles = []
    for l1 in left:
        for l2 in left:
            if l1 >= l2: continue
            for r1 in right:
                for r2 in right:
                    if r1 >= r2: continue
                    potential_cycles.append([(l1,r1), (l2,r1), (l2,r2), (l1,r2)])
    
    def is_bad_cycle(cycle_edges, edges_dict):
        # Check if all edges exist
        for e in cycle_edges:
            if e not in edges_dict:
                return False
        colors = [edges_dict[e] for e in cycle_edges]
        return len(set(colors)) != 4

    def get_bad_cycles_list(edges_dict):
        bads = []
        for cyc in potential_cycles:
            if is_bad_cycle(cyc, edges_dict):
                bads.append(cyc)
        return bads

    # Start search
    bad_cycles_list = get_bad_cycles_list(edges_dict)
    
    max_iter = 5000
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        if not bad_cycles_list:
            # Valid coloring found for current graph
            current_count = len(edges_dict)
            if current_count > best_count:
                best_count = current_count
                best_edges = [[u, v, c] for (u, v), c in edges_dict.items()]
            
            # Try to add more edges?
            # We are at K_{10,10}. We can't add more bipartite edges.
            # We could try to add edges within left or right?
            # That would create triangles, and 4-cycles involving those.
            # Let's stick to bipartite for now to ensure speed and validity.
            # If we have time, we can try non-bipartite later.
            # For now, break if we have a full K_{10,10} valid.
            if current_count == a * b:
                break
            else:
                # If we removed edges to get valid, try to add them back?
                # We don't track removed edges easily here.
                # Let's just restart with a different random seed if we are stuck at low density?
                # Or just continue searching for better coloring of current graph?
                # If bad_cycles_list is empty, we are optimal for this edge set.
                # We should try a different edge set.
                # Let's break and restart with a different random initialization.
                break

        # Pick a random bad cycle
        bad_cyc = random.choice(bad_cycles_list)
        
        # Decide: Recolor or Remove
        # If we have many edges, recoloring is preferred.
        # If we have few, removing might be necessary.
        # Let's say 70% recolor, 30% remove.
        if random.random() < 0.7:
            # Pick a random edge from the cycle
            e = random.choice(bad_cyc)
            old_color = edges_dict[e]
            
            best_new_color = old_color
            best_bad_count = len(bad_cycles_list)
            
            # Try all colors
            for c in range(20):
                if c == old_color:
                    continue
                edges_dict[e] = c
                # Recalculate bad cycles
                # Optimization: Only check cycles involving e?
                # For simplicity, recalculate all.
                # To speed up, we can just count.
                new_bad_count = 0
                for cyc in potential_cycles:
                    if is_bad_cycle(cyc, edges_dict):
                        new_bad_count += 1
                if new_bad_count < best_bad_count:
                    best_bad_count = new_bad_count
                    best_new_color = c
            
            edges_dict[e] = best_new_color
            if best_new_color == old_color:
                # No improvement, maybe remove?
                pass
            else:
                # Update bad_cycles_list
                bad_cycles_list = get_bad_cycles_list(edges_dict)
        else:
            # Remove a random edge from the cycle
            e = random.choice(bad_cyc)
            del edges_dict[e]
            adj[bad_cyc[0][0]].discard(bad_cyc[0][1]) # Just for tracking, not strictly needed if we use edges_dict
            bad_cycles_list = get_bad_cycles_list(edges_dict)
            
            # Update best if this state is better?
            # We only update best if valid.
    
    # After loop, if we didn't break with a valid graph, check current state
    if not bad_cycles_list:
        current_count = len(edges_dict)
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v, c] for (u, v), c in edges_dict.items()]
            
    # If we didn't find a good solution, try a simpler fallback
    # E.g. a cycle C20
    if best_count < 20:
        for i in range(n):
            u = i
            v = (i + 1) % n
            c = i % 20
            best_edges.append([u, v, c])
            best_count = n
            
    return best_edges
