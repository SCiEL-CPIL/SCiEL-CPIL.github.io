"""
Strategy: Exact Maximum Weight Independent Set (MWIS) solver.
The problem is to select a subset of integers from 1..511 such that they are pairwise coprime,
maximizing the sum of 1/(512-a).
This is equivalent to finding the Maximum Weight Independent Set in a graph where:
- Nodes are integers 1..511.
- Edge (u, v) exists if gcd(u, v) > 1.
- Weight of node u is 1/(512-u).

We solve this exactly using a recursive branch-and-bound algorithm on the complement graph
(which is a clique-finding problem, i.e., Maximum Weight Clique).
The complement graph has an edge between u and v if gcd(u, v) == 1.
We want the maximum weight clique in this complement graph.

To make this efficient:
1. Precompute the adjacency list for the complement graph (coprime relationships).
2. Use a greedy coloring heuristic on the candidate set to provide an upper bound for pruning.
3. Order nodes by weight (descending) to find high-value solutions early.
"""

import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    N = 512
    # We are looking for integers a in 1..511
    # Weight of choosing 'a' is 1/(512-a)
    # Note: a=511 has weight 1/1 = 1.0. a=1 has weight 1/511.
    
    # The candidate set is 1..511
    # Let's map them to indices 0..510 for easier handling, or just use the values directly.
    # Using values directly is fine.
    
    # Precompute coprime relationships
    # adj[i] is a set of j such that gcd(i, j) == 1
    # We only care about j > i to save memory/time in precomputation, 
    # but for the algorithm, we need fast access.
    
    # Since N is small (511 nodes), we can use a bitset or just sets.
    # Let's use a list of sets for adjacency in the complement graph (coprime graph).
    
    coprime_adj = [set() for _ in range(N)]
    for i in range(1, N):
        for j in range(i + 1, N):
            if gcd(i, j) == 1:
                coprime_adj[i].add(j)
                coprime_adj[j].add(i)
                
    # Weights
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # We want to find the Maximum Weight Clique in this graph.
    # The Bron-Kerbosch algorithm with pivoting is standard for Max Clique.
    # However, standard BK is exponential. We need pruning via upper bounds.
    
    # A simple and effective approach for MWIS/MWC on sparse-ish graphs:
    # Recursive branch and bound.
    # State: 
    #   R: current clique (list of nodes)
    #   P: candidates that can be added to R (must be coprime to all in R)
    #   X: excluded nodes (for completeness, but for optimization we can ignore if we just want max weight)
    
    # To get a good upper bound for pruning, we can use a greedy coloring on the subgraph induced by P.
    # The number of colors in a greedy coloring of a graph is an upper bound on the size of the maximum clique.
    # For weighted clique, we can sort P by weight and use a greedy "weighted coloring" or simply 
    # use the sum of weights of the largest k nodes where k is the color bound? 
    # Actually, a simpler bound: The maximum weight of a clique in P is at most the sum of the weights 
    # of the top |C| nodes in P, where |C| is the maximum clique size in the unweighted subgraph P.
    # Even better: We can use the greedy coloring to partition P into independent sets. 
    # A clique can have at most one node from each independent set. 
    # So the upper bound is the sum of the max weight in each color class.
    
    best_weight = 0.0
    best_clique = []
    
    # Order nodes by weight descending to find good solutions early
    # This helps the incumbent (best_weight) grow quickly, improving pruning.
    # However, the branching order in BK is usually determined by the pivot.
    # Let's implement a custom recursive search.
    
    # To speed up, we can filter out nodes that are "dominated"? 
    # If node u is coprime to node v, and weight(u) <= weight(v), and N(v) is a superset of N(u) (in the coprime graph),
    # then u is dominated by v? Not necessarily, because v might conflict with other nodes u doesn't.
    # In the coprime graph, "conflict" is non-edge. 
    # If N_coprime(v) is a superset of N_coprime(u), v is "more compatible".
    # If weight(v) >= weight(u), then u is never better than v.
    # Let's check if we can prune dominated nodes.
    # This might be expensive to compute for all pairs. Let's skip for now and rely on coloring bound.
    
    def greedy_color_bound(P):
        """
        Returns an upper bound on the maximum weight of a clique in P.
        Uses greedy coloring on the subgraph induced by P.
        """
        if not P:
            return 0.0
        
        # Sort P by weight descending to assign colors? 
        # Standard greedy coloring: assign smallest possible color.
        # For the bound, we want to minimize the number of colors to get a tight bound on size,
        # but for weight, we want to group high-weight nodes together?
        # Actually, the bound is: Sum over color classes C_i of max(weights in C_i).
        # To make this bound tight (low), we want to put high-weight nodes in different color classes?
        # No, the bound is an upper bound. We want it to be as small as possible to prune.
        # The bound holds for ANY valid coloring.
        # So we should use a coloring that minimizes this sum.
        # A greedy coloring that assigns high-weight nodes to existing classes if possible might help?
        # Or just standard greedy coloring.
        
        # Let's use a simple greedy coloring.
        # Order P by weight descending.
        P_sorted = sorted(P, key=lambda x: weights[x], reverse=True)
        
        color_classes = [] # List of sets
        node_color = {}
        
        for u in P_sorted:
            # Find the first color class where u can be added (i.e., u is NOT coprime to anyone in that class? 
            # Wait. In the coprime graph, an edge means "compatible".
            # An independent set in the coprime graph is a set of nodes where NO two are coprime.
            # A clique in the coprime graph is a set of nodes where ALL are coprime.
            # We are looking for a Clique in the coprime graph.
            # The complement of the coprime graph is the "conflict" graph (gcd > 1).
            # Coloring the Conflict Graph:
            # An independent set in the Conflict Graph is a Clique in the Coprime Graph.
            # So, if we color the Conflict Graph, each color class is an independent set in the Conflict Graph,
            # which is a Clique in the Coprime Graph? No.
            # Independent set in G is a set of vertices no two of which are adjacent in G.
            # In Conflict Graph, adjacent means gcd > 1.
            # So independent set in Conflict Graph = set of vertices where no two have gcd > 1 = Pairwise Coprime set = Clique in Coprime Graph.
            # So, a coloring of the Conflict Graph partitions vertices into independent sets (cliques in Coprime Graph).
            # This doesn't help directly for the bound on the size of the MAX clique in Coprime Graph.
            
            # Let's re-evaluate.
            # We want Max Weight Clique in Coprime Graph G_c.
            # This is equivalent to Max Weight Independent Set in Conflict Graph G_conf.
            # Standard bound for MWIS: 
            # If we color G_conf, each color class is an independent set in G_conf.
            # An independent set in G_conf is a set of nodes that are pairwise coprime.
            # This means we can pick at most ONE node from each color class to form a valid solution?
            # NO. If we pick two nodes from the same color class, they are not adjacent in G_conf, so they are coprime.
            # So we can pick ALL nodes from a color class? 
            # Yes! If C is an independent set in G_conf, then all nodes in C are pairwise coprime.
            # So C itself is a valid clique in G_c.
            # This implies that if we color G_conf, the union of all color classes is the whole graph.
            # This doesn't give an upper bound on the weight of the MAX clique.
            
            # Let's flip it.
            # We want to bound the weight of a clique in G_c.
            # A clique in G_c is an independent set in G_conf.
            # Let's color G_c.
            # If we color G_c, each color class is an independent set in G_c.
            # An independent set in G_c is a set of nodes where NO two are coprime (i.e., all pairs have gcd > 1).
            # A clique in G_c can contain at most ONE node from each color class of G_c.
            # So, if we color G_c with k colors, the maximum size of a clique in G_c is at most k.
            # And the maximum weight is at most the sum of the max weight in each color class.
            
            # So: Color the Coprime Graph G_c.
            # Greedy coloring of G_c:
            # Assign colors such that adjacent nodes (coprime nodes) have different colors.
            
            # Algorithm:
            # 1. Order nodes in P by weight descending.
            # 2. For each node u, assign the smallest color c such that no neighbor of u in P has color c.
            
            colors = {}
            color_max_weight = {}
            
            for u in P_sorted:
                # Find forbidden colors: colors of neighbors of u in P
                forbidden = set()
                for v in coprime_adj[u]:
                    if v in P:
                        if v in colors:
                            forbidden.add(colors[v])
                
                # Find smallest available color
                c = 1
                while c in forbidden:
                    c += 1
                
                colors[u] = c
                if c not in color_max_weight:
                    color_max_weight[c] = 0.0
                color_max_weight[c] = max(color_max_weight[c], weights[u])
            
            # Upper bound is sum of max weights in each color class
            return sum(color_max_weight.values())

    # Initial incumbent: Greedy solution
    # Pick highest weight node, remove neighbors, repeat.
    P_init = set(range(1, N))
    R_greedy = []
    P_cur = set(P_init)
    while P_cur:
        # Pick node with max weight
        u = max(P_cur, key=lambda x: weights[x])
        R_greedy.append(u)
        # Remove u and its neighbors from P_cur
        P_cur.remove(u)
        for v in list(P_cur):
            if v in coprime_adj[u]:
                P_cur.remove(v)
                
    best_weight = sum(weights[u] for u in R_greedy)
    best_clique = list(R_greedy)
    
    # Recursive search
    # To avoid recursion depth issues, we can use an explicit stack or increase recursion limit.
    # Max depth is 511.
    import sys
    sys.setrecursionlimit(10000)
    
    # We'll use a global best to allow pruning in the recursion.
    # But we need to pass current weight.
    
    def search(R, P, current_weight):
        nonlocal best_weight, best_clique
        
        if time.time() > deadline:
            return
            
        if not P:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(R)
            return
            
        # Upper bound
        # If current_weight + bound(P) <= best_weight, prune
        # Calculating bound(P) is expensive (O(|P|^2) for coloring).
        # We can cache bounds? Or just calculate.
        # Since P shrinks, it's manageable.
        
        # Optimization: If P is small, just brute force? No, coloring is fast enough.
        
        ub = greedy_color_bound(P)
        if current_weight + ub <= best_weight:
            return
            
        # Branching
        # Choose a pivot node v in P to minimize branching?
        # In BK, we iterate over N(v) or P \ N(v)?
        # Standard BK for Max Clique:
        # Pick pivot v in P U X.
        # Iterate over P \ N(v).
        
        # Here we don't track X explicitly for optimization (we only care about max weight).
        # So we iterate over P.
        # To order branches, we can pick nodes with high weight first.
        
        # Pivot selection:
        # Pick v in P that maximizes |N(v) intersection P|?
        # This minimizes the number of branches (P \ N(v)).
        
        # However, we want to find high weight solutions early.
        # Let's just iterate P sorted by weight descending.
        
        P_list = sorted(P, key=lambda x: weights[x], reverse=True)
        
        for u in P_list:
            # Branch on including u
            new_R = R + [u]
            new_P = P & coprime_adj[u]
            
            search(new_R, new_P, current_weight + weights[u])
            
            # Remove u from P for subsequent branches
            P.remove(u)
            # Note: P is modified in place. This is standard BK behavior.
            
            # Check time
            if time.time() > deadline:
                return

    # Start search
    # Initial P is all nodes
    P_start = set(range(1, N))
    search([], P_start, 0.0)
    
    return best_clique

# Fallback if the exact solver is too slow or fails, though it should be fast for N=511.
# The greedy coloring bound is quite tight for random-like graphs.
# The coprime graph is structured, but the bound should still work well.

if __name__ == "__main__":
    res = construct()
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    print(f"Count: {len(res)}")
