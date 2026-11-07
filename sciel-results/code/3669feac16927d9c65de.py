"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split_graph(n, k, p, seed):
        if time.time() > deadline:
            edges = instance(n, k, p, seed)
            return [[u, v] for u, v in edges]
            
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # 1. Parse Graph Structure
        E_CC = set()
        E_CI = {} # v in I -> set of neighbors in C
        
        for u, v in edges:
            if u < k and v < k:
                E_CC.add((u, v))
            elif u < k and v >= k:
                if v not in E_CI:
                    E_CI[v] = set()
                E_CI[v].add(u)
            elif v < k and u >= k:
                if u not in E_CI:
                    E_CI[u] = set()
                E_CI[u].add(v)
        
        # Pre-compute all possible cliques of size >= 2
        # We only need to consider:
        # 1. Type A: Subsets of C (cliques in C)
        # 2. Type B: {v} U S where v in I and S is a clique in N(v)
        
        # For efficiency, we will generate candidates on the fly or pre-compute small ones.
        # Given k <= 30, the number of subsets is large, but we only need maximal ones or specific sizes.
        # However, Lagrangian relaxation requires evaluating "value" of a clique.
        
        # Let's use a heuristic iterative approach:
        # 1. Initialize weights w_e = 1 for all edges.
        # 2. Loop:
        #    a. Find a set of edge-disjoint cliques maximizing sum of weights of covered edges.
        #       (Approximated by greedy: pick clique with highest weight sum, remove its edges, repeat)
        #    b. If all edges covered, stop.
        #    c. Update weights: increase weight of uncovered edges to force them to be picked next time.
        
        all_edges = list(E_CC) + [(u, v) for v in E_CI for u in E_CI[v]]
        edge_set = set(all_edges)
        
        # Weights for Lagrangian
        weights = {e: 1.0 for e in edge_set}
        
        best_cliques = None
        best_count = float('inf')
        
        max_iter = 50 # Limit iterations to fit in time
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Greedy Packing with Weighted Objective
            current_cliques = []
            remaining_edges = set(edge_set)
            
            # To speed up, we generate candidate cliques based on current "pressure" (weights)
            # We prioritize cliques that cover high-weight edges.
            
            # Heuristic: 
            # 1. Identify "bottleneck" edges (high weight).
            # 2. Try to form large cliques around these edges.
            
            # Let's simplify: 
            # We will greedily pick the best available clique.
            # A clique is "available" if all its edges are in remaining_edges.
            # Value of clique = sum(weights[e] for e in clique_edges) - lambda * (1 if size==2 else 0)
            # Actually, we want to minimize number of cliques. 
            # The Lagrangian dual approach usually maximizes covered weight.
            # If we cover all edges, the number of cliques is the count.
            
            # Let's try a different angle: 
            # The "cost" of a clique is 1. The "benefit" is covering edges.
            # We want to cover all edges with min cost.
            # This is Set Cover.
            # Lagrangian relaxation of "cover all edges" constraint:
            # Minimize sum(x_K) + sum(lambda_e * (1 - sum_{K covers e} x_K))
            # = sum(x_K * (1 - sum_{e in K} lambda_e)) + sum(lambda_e)
            # To minimize, we pick cliques K where sum_{e in K} lambda_e > 1.
            # Then we update lambda_e += alpha * (1 - covered_e).
            
            # Let's implement this specific heuristic:
            # 1. Start with lambda_e = 0.
            # 2. In each step, find a clique K such that sum_{e in K} lambda_e > 1 (or max sum if none).
            #    Actually, if all lambdas are 0, no clique has sum > 1.
            #    So we must start with some initialization or different update rule.
            
            # Revised Heuristic:
            # We want to minimize count. 
            # Greedy: Pick clique that covers the most "hard" edges.
            # "Hardness" is tracked by lambda.
            
            # Let's restart the loop logic properly.
            
            remaining_edges = set(edge_set)
            current_cliques = []
            
            # We will repeatedly pick the "best" clique from remaining_edges.
            # To avoid O(2^k) search, we use a constructive greedy:
            # 1. Pick an uncovered edge (u, v) with highest lambda.
            # 2. Try to extend {u, v} to a maximal clique using other uncovered edges.
            # 3. Add this clique to current_cliques.
            # 4. Remove its edges from remaining_edges.
            
            while remaining_edges:
                if time.time() > deadline:
                    # Fallback: cover remaining with edges
                    for e in remaining_edges:
                        current_cliques.append([e[0], e[1]])
                    remaining_edges.clear()
                    break
                
                # Pick edge with max lambda
                max_lam = -1
                target_edge = None
                for e in remaining_edges:
                    if weights[e] > max_lam:
                        max_lam = weights[e]
                        target_edge = e
                
                if target_edge is None:
                    break
                
                u, v = target_edge
                
                # Try to extend {u, v} to a larger clique using remaining_edges
                # Candidates are neighbors of u and v in the graph induced by remaining_edges?
                # No, the clique must be valid in the original graph.
                # But all its edges must be in remaining_edges.
                
                # Let's find common neighbors w such that (u,w) and (v,w) are in remaining_edges.
                # Then check if (w, w2) etc.
                
                # Simple Greedy Extension:
                # Start with S = {u, v}
                # Find all w such that (min(u,w),max(u,w)) in remaining_edges AND (min(v,w),max(v,w)) in remaining_edges
                # Then iteratively add w that maintains clique property w.r.t remaining_edges.
                
                S = [u, v]
                S_set = {u, v}
                
                # Get potential candidates
                # For each w in range(n):
                #   if w not in S_set:
                #     if all (min(x,w),max(x,w)) in remaining_edges for x in S:
                #       add to candidates
                
                candidates = []
                for w in range(n):
                    if w in S_set:
                        continue
                    valid = True
                    for x in S:
                        e = (min(x, w), max(x, w))
                        if e not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        candidates.append(w)
                
                # Sort candidates by degree in remaining_edges (within S U candidates) to pick "dense" ones first?
                # Or just pick one that adds the most value?
                # Let's just greedily add candidates that keep S a clique in remaining_edges.
                
                # To maximize size, we can try to add as many as possible.
                # This is still Max Clique on the induced subgraph of remaining_edges.
                # Given small k, we can do a simple backtracking or greedy.
                
                # Greedy:
                while True:
                    added = False
                    # Try adding candidates in order of "connectivity"
                    # Sort candidates by number of connections to S
                    def score(w):
                        cnt = 0
                        for x in S:
                            e = (min(x, w), max(x, w))
                            if e in remaining_edges:
                                cnt += 1
                        return cnt
                    
                    candidates.sort(key=score, reverse=True)
                    
                    for w in candidates:
                        # Check if w can be added to S
                        # It must be connected to all existing members of S via remaining_edges
                        # (This is already true by construction of candidates, but S might have changed? No, S is fixed in this loop step)
                        # Wait, if we add w1, then w2 must be connected to w1 too.
                        # So we need to check against current S.
                        
                        can_add = True
                        for x in S:
                            e = (min(x, w), max(x, w))
                            if e not in remaining_edges:
                                can_add = False
                                break
                        
                        if can_add:
                            S.append(w)
                            S_set.add(w)
                            # Remove w from candidates
                            candidates.remove(w)
                            # Update other candidates: they must now be connected to w
                            new_candidates = []
                            for w2 in candidates:
                                e = (min(w, w2), max(w, w2))
                                if e in remaining_edges:
                                    new_candidates.append(w2)
                            candidates = new_candidates
                            added = True
                            break # Restart loop with new S
                            
                    if not added:
                        break
                
                # Add S as a clique
                if len(S) >= 2:
                    current_cliques.append(S)
                    # Remove edges in S from remaining_edges
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            remaining_edges.discard(e)
            
            # After greedy packing, check if we covered everything
            if not remaining_edges:
                if len(current_cliques) < best_count:
                    best_count = len(current_cliques)
                    best_cliques = current_cliques
                break # Success
            
            # Update weights for next iteration
            # Increase weight of uncovered edges
            for e in remaining_edges:
                weights[e] += 1.0
            # Decay weights of covered edges? Or keep them?
            # If we keep them, they won't be picked again because they are not in remaining_edges.
            # But for the next iteration, we restart from scratch with new weights.
            
            # To prevent infinite loop or poor convergence, we might need to randomize or restart weights occasionally.
            # For now, simple additive update.
            
        # If best_cliques is None or didn't cover all (should have in fallback), use fallback
        if best_cliques is None:
            # Fallback: cover all edges with size-2 cliques
            best_cliques = [[u, v] for u, v in edge_set]
            
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
        else:
            res = solve_split_graph(*spec)
            results.append(res)
            
    return results
# EVOLVE-BLOCK-END
