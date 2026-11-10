"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    # Precompute conflict graph for 4-APs
    # conflict[i] is a set of j such that i and j cannot be in the same set
    # This happens if there exists a, b, c in [1..n] such that {i, j, a, b} forms a 4-AP
    # Or more simply, we can just check if adding j to a set containing i creates a 4-AP.
    # But for MIS, we need a static graph where edge (i,j) exists if i and j are in a 4-AP.
    # Wait, if i and j are in a 4-AP, they cannot both be in the independent set.
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # Any pair of elements in this set conflicts.
    
    conflicts = [set() for _ in range(n + 1)]
    
    # Generate all 4-APs
    for a in range(1, n + 1):
        # d such that a+3d <= n
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            terms = [a, a + d, a + 2 * d, a + 3 * d]
            # Add conflicts between all pairs in this AP
            # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = terms[i], terms[j]
                    conflicts[u].add(v)
                    conflicts[v].add(u)
    
    # Convert to lists for faster access if needed, but sets are fine for membership.
    # We will use a branch and bound algorithm for Maximum Independent Set.
    # This is equivalent to Maximum Clique on the complement graph, but let's stick to MIS on conflict graph.
    
    # To make B&B efficient, we use the "Tomita" style greedy coloring bound.
    # However, implementing a full robust Tomita in Python within 85s for N=512 is risky.
    # Let's try a simpler B&B with greedy coloring of the candidate set.
    
    # We will represent the set of candidates as a list of nodes.
    # The bound is the size of the maximum clique in the subgraph induced by candidates,
    # which is an upper bound on the MIS size? No.
    # MIS on G is Max Clique on G_complement.
    # Let's work on the complement graph for Max Clique.
    # Node i is connected to j in G_comp if i and j are NOT in conflict.
    # We want Max Clique in G_comp.
    
    # Precompute non-conflicts (complement edges)
    # Actually, checking non-conflict is O(1) with a boolean matrix or set.
    # Let's use a bitset or boolean matrix for G_comp.
    # N=512, matrix is 512x512 = 262k booleans. Fine.
    
    # adj[i][j] is True if i and j are compatible (not in conflict)
    # Note: i and i are compatible (trivially), but clique vertices are distinct.
    adj = [[False] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        adj[i][i] = True
        for j in range(1, n + 1):
            if j != i and j not in conflicts[i]:
                adj[i][j] = True
    
    # Greedy Coloring for Bound
    # We color the candidate set such that adjacent nodes in G_comp (compatible nodes)
    # wait, for Max Clique, we color the *candidate subgraph*.
    # The bound is the number of colors needed to color the candidate graph.
    # If we use k colors, the max clique size is at most k.
    # We color greedily.
    
    def color_sort(P, adj, n):
        """
        P is a list of candidate nodes.
        Returns (P_sorted, colors) where P_sorted is ordered by color (descending),
        and colors[i] is the color of P_sorted[i].
        """
        if not P:
            return [], []
        
        # Unassigned nodes
        unassigned = P[:]
        colors = [0] * len(unassigned)
        P_sorted = [0] * len(unassigned)
        
        # Greedy coloring
        # We want to minimize colors, but for bound, any coloring works.
        # Standard Tomita sorts by color class size or just appends.
        
        color_count = 0
        # We will assign colors 1, 2, ...
        # For each node, find the lowest color it can take.
        
        # This simple O(N^2) coloring is fine for N=512.
        # Actually, we want to order the nodes for B&B.
        # The ordering is usually by color.
        
        # Let's implement a standard greedy coloring that returns the color of each node.
        # Then sort nodes by color.
        
        node_colors = {}
        available_colors = [False] * (len(unassigned) + 1) # 1-indexed colors
        
        # To speed up, we can just assign 1, 2, 3...
        # But we need a valid coloring.
        
        # Simpler approach for bound:
        # We just need an upper bound.
        # A trivial upper bound is len(P).
        # A better one is the chromatic number.
        
        # Let's do a simple greedy coloring.
        # Order nodes by degree in P (descending) to get fewer colors?
        # Or just arbitrary?
        # Let's try arbitrary first.
        
        for idx, node in enumerate(unassigned):
            # Check which colors are forbidden
            # Forbidden if neighbor in P has that color
            forbidden = [False] * (len(unassigned) + 1)
            for jdx, neighbor in enumerate(unassigned):
                if neighbor == node:
                    continue
                if adj[node][neighbor]:
                    if node_colors.get(neighbor, 0) > 0:
                        forbidden[node_colors[neighbor]] = True
            
            # Find smallest available color
            c = 1
            while forbidden[c]:
                c += 1
            node_colors[node] = c
            if c > color_count:
                color_count = c
        
        # Create list of (node, color)
        colored_nodes = [(node, node_colors[node]) for node in unassigned]
        
        # Sort by color descending, then by node id?
        # Tomita sorts by color.
        colored_nodes.sort(key=lambda x: (-x[1], x[0]))
        
        P_sorted = [x[0] for x in colored_nodes]
        colors = [x[1] for x in colored_nodes]
        
        return P_sorted, colors, color_count

    best_clique = []
    best_size = 0
    
    # Initial greedy clique to get a good lower bound
    # Greedy Max Clique heuristic
    P_init = list(range(1, n + 1))
    random.shuffle(P_init) # Randomize to break symmetry? Or use natural order?
    # Let's use a greedy heuristic to find a large clique
    current_clique = []
    P_curr = P_init[:]
    while P_curr:
        # Pick node with max degree in current subgraph? Or just first?
        # Let's pick node with max connections to existing clique + future?
        # Simple greedy: pick node that connects to all in current_clique.
        # To maximize size, we should pick nodes that allow more future nodes.
        # Let's just pick the first valid node.
        
        valid_node = None
        for node in P_curr:
            if all(adj[node][c] for c in current_clique):
                valid_node = node
                break
        
        if valid_node is None:
            break
        
        current_clique.append(valid_node)
        P_curr.remove(valid_node)
        
        # Remove nodes that are not connected to valid_node
        P_curr = [x for x in P_curr if adj[x][valid_node]]
        
    if len(current_clique) > best_size:
        best_size = len(current_clique)
        best_clique = current_clique[:]
        
    # B&B
    # We expand from the greedy solution? Or from scratch?
    # Standard Tomita starts with empty clique and P = all nodes.
    
    # To save time, we can start with the greedy clique as a base?
    # No, B&B explores the tree.
    
    # Let's implement the recursive B&B.
    # To avoid deep recursion issues, we might need to be careful.
    # N=512, depth can be 512. Python recursion limit is 1000. OK.
    
    # We will use a global best_size to prune.
    
    def expand(C, P, colors, color_count):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return False # Signal timeout
            
        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = C[:]
            return True
            
        # P is sorted by color descending.
        # colors[i] is the color of P[i].
        # The max clique in P is at most colors[0] (the highest color).
        # Actually, the bound is the color of the first node in the sorted list?
        # No, the bound is the maximum color used.
        # But we can prune if len(C) + colors[0] <= best_size.
        
        if len(C) + colors[0] <= best_size:
            return True
            
        # Iterate through P in reverse order (highest color first)
        # Standard Tomita iterates i from 0 to len(P)-1
        # But since P is sorted by color descending, P[0] has highest color.
        # We should try P[0] first?
        # Actually, Tomita iterates i from 0 to |P|-1.
        # If we pick P[i], we recurse with P[0..i-1] (nodes with color >= color[i] that are before i? No.)
        # The property is: if we pick P[i], the next candidates must be from P[0..i-1]
        # because P[i+1..] have lower or equal color, and we process in order.
        # Wait, the standard algorithm:
        # for i in range(len(P)):
        #   if len(C) + colors[i] <= best_size: return
        #   expand(C + [P[i]], P[:i], ...)
        
        for i in range(len(P)):
            if time.time() > deadline:
                return False
                
            if len(C) + colors[i] <= best_size:
                return True # Since colors are sorted descending, all subsequent i' > i have colors[i'] <= colors[i]
            
            # Add P[i] to clique
            new_C = C + [P[i]]
            
            # New P is P[0..i-1] filtered by neighbors of P[i]
            # P[:i] contains nodes with index < i.
            # We need to keep only those connected to P[i].
            new_P = [P[j] for j in range(i) if adj[P[i]][P[j]]]
            
            if new_P:
                # Recolor new_P
                # The colors of new_P must be recomputed relative to the subgraph.
                # However, the existing colors are an upper bound on the clique size in that subgraph.
                # So we can just use the existing colors?
                # Yes, the color of a node in a subgraph is <= color in the supergraph.
                # So colors[j] for j < i are valid upper bounds.
                # We just need to filter the colors list to match new_P.
                new_colors = [colors[j] for j in range(i) if adj[P[i]][P[j]]]
                
                # Note: new_P and new_colors are aligned.
                # We don't need to sort again if we preserve relative order?
                # The order in P was by color descending.
                # P[:i] is also by color descending.
                # Filtering preserves order.
                # So new_P is sorted by color descending.
                
                expand(new_C, new_P, new_colors, colors[0])
            else:
                # Leaf
                if len(new_C) > best_size:
                    best_size = len(new_C)
                    best_clique = new_C[:]
                    
    # Initial call
    # P is all nodes 1..n
    P0 = list(range(1, n + 1))
    # Sort P0 by degree descending? Or just use natural order?
    # Tomita suggests ordering by degree.
    # Let's order by degree in the complement graph (number of non-conflicts).
    # Actually, for Max Clique, we want high degree.
    deg = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and adj[i][j]:
                deg[i] += 1
                
    P0.sort(key=lambda x: -deg[x])
    
    P0_sorted, colors0, max_color0 = color_sort(P0, adj, n)
    
    # Wait, color_sort returns P_sorted.
    # We should use P0_sorted for the initial P.
    
    expand([], P0_sorted, colors0, max_color0)
    
    # If we timed out, best_clique might be empty or partial.
    # We need to ensure we return a valid set.
    # The greedy solution was found before B&B.
    # If B&B didn't update best_clique, we use the greedy one.
    
    if not best_clique:
        # Fallback: return the greedy solution found earlier
        # We stored it in best_clique? No, we updated best_clique if greedy was better.
        # Let's re-verify.
        # The greedy loop updated best_clique.
        # If B&B found nothing better, best_clique is the greedy one.
        pass
    
    # The result is the set of nodes in best_clique.
    # These nodes form a clique in G_comp, which is an independent set in G (conflict graph).
    # So they contain no 4-AP.
    
    result = sorted(best_clique)
    
    # Final sanity check: ensure non-empty
    if not result:
        # Extremely unlikely, but return a single element
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
