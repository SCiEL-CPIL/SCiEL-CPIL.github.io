import time
from math import gcd, lcm
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    max_n = 512
    
    # 1. Build compatibility graph and adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if i and j are compatible (lcm(i,j) <= 512)
    adj = [0] * (max_n + 1)
    
    for i in range(1, max_n + 1):
        for j in range(i + 1, max_n + 1):
            if lcm(i, j) <= max_n:
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Maximum Clique Search using Tomita's Algorithm with Greedy Coloring
    # We work on the subgraph of nodes 1..max_n
    # Nodes are 1-indexed in our logic, but bitmasks are 0-indexed bits?
    # Let's use 0-indexed bits for simplicity in bit ops, mapping node k to bit k-1.
    # Or simpler: keep node indices 1..N, and use bit k to represent node k.
    # Python ints can handle bit 512 easily.
    
    best_clique = []
    best_size = 0
    
    # Greedy coloring for pruning
    # Returns list of colors for nodes in P, and the max color
    # We also need to order nodes by color to help the search
    def greedy_coloring(P):
        """
        P is a set of nodes (integers).
        Returns a list of tuples (node, color) sorted by color, and the max color.
        """
        if not P:
            return [], 0
        
        colors = {}
        uncolored = set(P)
        color = 0
        
        # Standard greedy coloring: assign smallest possible color
        # To optimize clique search, we usually color in an order that produces
        # a good upper bound. One common heuristic is to order by degree.
        # However, for the coloring bound in MaxClique, we just need a valid coloring
        # of the subgraph induced by P.
        
        # To get a tighter bound, we can sort P by degree in the original graph
        # or in the subgraph P. Let's sort by degree in the full graph first.
        sorted_P = sorted(P, key=lambda x: bin(adj[x] & (sum([1<<y for y in P])).__int__()).count('1'), reverse=True)
        
        # Actually, computing degree in P is expensive if done naively.
        # Let's just use the standard greedy coloring on the set.
        # A better heuristic for coloring in MaxClique is to pick nodes in an order
        # that minimizes the number of colors.
        
        # Simple greedy:
        # We maintain a list of available colors for each node?
        # No, just assign the first available color.
        
        # To make the bound tighter, we often color the nodes in the order they are
        # added to the candidate set.
        
        # Let's use a specific ordering: sort by degree in the subgraph P.
        # deg_P(x) = popcount(adj[x] & mask(P))
        
        mask_P = 0
        for x in P:
            mask_P |= (1 << x)
            
        def popcount(x):
            return bin(x).count('1')
            
        # Sort by degree in P descending
        sorted_nodes = sorted(list(P), key=lambda x: popcount(adj[x] & mask_P), reverse=True)
        
        color_of_node = {}
        current_color = 0
        uncolored_nodes = list(sorted_nodes)
        
        # This is a simple greedy coloring. 
        # For better performance in MaxClique, we often use a "color sort" where
        # we assign colors such that nodes with higher colors are considered later?
        # No, Tomita's algorithm processes nodes in order of increasing color.
        
        # Let's implement the standard coloring used in Bron-Kerbosch with pivoting
        # or Tomita's DSATUR-like coloring.
        
        # Simpler approach for Python:
        # Just color greedily.
        assigned = {}
        avail_colors = {}
        
        # We will assign colors 1, 2, ...
        # To get a good bound, we want the max color to be small.
        
        # Let's iterate through sorted_nodes
        for node in sorted_nodes:
            # Find smallest color not used by neighbors in assigned set
            used_colors = set()
            neighbors_in_assigned = [n for n in assigned if (adj[node] & (1 << n))]
            for n in neighbors_in_assigned:
                used_colors.add(assigned[n])
            
            c = 1
            while c in used_colors:
                c += 1
            
            assigned[node] = c
            if c > current_color:
                current_color = c
        
        # Return list of (node, color) sorted by color
        # Nodes with same color can be in any order
        result = sorted([(n, c) for n, c in assigned.items()], key=lambda x: x[1])
        return result, current_color

    def expand(P, clique):
        """
        P is a set of nodes compatible with all nodes in clique.
        clique is the current list of nodes in the clique.
        """
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if not P:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        
        # Coloring
        colored_P, max_color = greedy_coloring(P)
        
        # If max_color + len(clique) <= best_size, prune
        if len(clique) + max_color <= best_size:
            return
        
        # Process nodes in order of increasing color
        # We maintain a set of remaining nodes
        remaining = set(P)
        
        # We iterate through the colored list
        # For each node u with color c:
        #   If len(clique) + c <= best_size: break (since remaining nodes have color >= c)
        #   Add u to clique
        #   Expand with P \ {v | v is not adjacent to u}
        #   Remove u from remaining
        
        # Note: The coloring is static for this call.
        # The condition "len(clique) + color(u) <= best_size" works because
        # all nodes processed after u have color >= color(u).
        
        # We need to iterate in the order of the colored list
        for node, color in colored_P:
            if node not in remaining:
                continue
                
            if len(clique) + color <= best_size:
                break
                
            # Add node to clique
            new_clique = clique + [node]
            
            # New P is intersection of P and neighbors of node
            # P_new = {v in P | v is adjacent to node}
            # We can compute this using bitmasks for speed
            mask_node = 1 << node
            mask_adj = adj[node]
            
            # Filter remaining nodes that are adjacent to node
            # But we also need to remove 'node' itself from the candidate set for the next level?
            # Yes, expand expects a set of candidates.
            
            # The standard recursive step:
            # expand(P \ {v | v not in N(u)}, clique U {u})
            # Here P is the set of candidates.
            
            # Let's construct the new set of candidates
            # It must be a subset of the previous P, and adjacent to node.
            # Also, we should process them in a way that respects the coloring bound?
            # The recursive call will re-color.
            
            # To be efficient, we can pass the set of neighbors in P.
            # P_new = {v in P | (adj[node] >> v) & 1}
            
            P_new = set()
            for v in P:
                if v == node:
                    continue
                if (adj[node] & (1 << v)):
                    P_new.add(v)
            
            expand(P_new, new_clique)
            
            # Remove node from remaining
            remaining.discard(node)

    # Start search
    # Initial P is all nodes
    initial_P = set(range(1, max_n + 1))
    expand(initial_P, [])
    
    if not best_clique:
        # Fallback to a simple valid set
        best_clique = [1]
        
    return list(best_clique)
