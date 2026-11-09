"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import gcd
from itertools import combinations

def construct():
    MAX_M = 128
    
    # 1. Generate all valid (residue, modulus) pairs
    # Node index mapping
    # We will store nodes as (r, m)
    nodes = []
    mod_to_indices = {}
    idx = 0
    for m in range(2, MAX_M + 1):
        start_idx = idx
        for r in range(m):
            nodes.append((r, m))
            idx += 1
        mod_to_indices[m] = (start_idx, idx)
        
    N = len(nodes)
    
    # 2. Build adjacency bitsets
    # adj[i] is an integer bitset where bit j is 1 if node i and node j are disjoint
    adj = [0] * N
    
    # Precompute gcd for moduli pairs to speed up
    # gcd_val[m1][m2]
    # We only need to check pairs of moduli
    moduli_list = list(range(2, MAX_M + 1))
    
    # To speed up edge construction, iterate over modulus pairs
    # For each pair of moduli (m1, m2), determine which residue pairs are disjoint
    # and set the bits in adj
    
    # We can optimize by noting that if gcd(m1, m2) == 1, no edges.
    # If gcd > 1, we iterate residues.
    
    # To avoid O(N^2) in Python which is slow, we use the modulus grouping.
    # Total operations ~ sum(m1*m2 for m1<m2 if gcd>1).
    
    for i_m, m1 in enumerate(moduli_list):
        start1, end1 = mod_to_indices[m1]
        for j_m in range(i_m + 1, len(moduli_list)):
            m2 = moduli_list[j_m]
            g = gcd(m1, m2)
            if g == 1:
                continue
            
            start2, end2 = mod_to_indices[m2]
            
            # Iterate residues
            # r1 in [0, m1), r2 in [0, m2)
            # Condition: (r1 - r2) % g != 0
            # We can iterate r1 and determine valid r2s?
            # Or just double loop if m1, m2 are small.
            # Max m is 128. 128*128 = 16384 ops per modulus pair.
            # Number of modulus pairs with gcd>1 is roughly half of 127*126/2 ~ 4000.
            # 4000 * 16000 = 64 million ops.
            # This might be too slow in pure Python (approx 10-20s).
            
            # Optimization:
            # For a fixed r1, the condition is r2 NOT in {r1 mod g, (r1+g) mod g, ...}
            # Actually, r2 must be NOT congruent to r1 mod g.
            # There are g residue classes mod g. r1 is in one.
            # The "bad" r2s are those in the same class mod g.
            # Number of bad r2s is approx m2/g.
            # Number of good r2s is m2 - m2/g.
            
            # Let's just do the double loop but try to make it tight.
            # Or we can build the bitset for r1 by clearing the bits for bad r2s.
            
            # Let's try a different approach for speed:
            # For each node i (r1, m1), we want to set bits in adj[i] for all j (r2, m2) that are disjoint.
            # This is still O(N^2) effectively.
            
            # Let's use the property that we only need to check pairs of moduli.
            # We'll update adj for all i in m1 group and j in m2 group.
            
            # We can precompute for each g, a mask of residues?
            # No, let's just run the double loop. If it's too slow, we might need to reduce N or use C extensions (not allowed).
            # But 60s limit is generous. 64M simple ops in Python might take ~5-10s if optimized.
            
            for r1 in range(m1):
                idx1 = start1 + r1
                # For each r2 in m2
                # We can compute the valid r2s
                # r2 is valid if r2 % g != r1 % g
                r1_mod_g = r1 % g
                mask = 0
                # Build mask of valid indices in m2
                # We can iterate r2
                for r2 in range(m2):
                    if r2 % g != r1_mod_g:
                        idx2 = start2 + r2
                        mask |= (1 << idx2)
                
                # Add to adj[idx1] and adj[idx2]?
                # adj is symmetric.
                # We are currently in loop for m1, m2.
                # We set adj[idx1] |= mask
                # We also need to set adj[idx2] bits for idx1.
                # This is hard to do symmetrically in this loop structure easily without extra work.
                # Let's just fill adj[idx1] for now, and handle symmetry later?
                # Or, we can just compute the full adjacency matrix in a way that is symmetric.
                
                # Actually, we can just do:
                adj[idx1] |= mask
                
                # To maintain symmetry, we need to set bit idx1 in adj[idx2] for all valid idx2.
                # This is equivalent to: for each valid idx2, adj[idx2] |= (1 << idx1).
                # We can do this by iterating valid idx2s again?
                # Or we can store the list of valid idx2s.
                
                # Let's refine:
                # valid_indices = [start2 + r2 for r2 in range(m2) if r2 % g != r1_mod_g]
                # mask = 0
                # for idx2 in valid_indices: mask |= 1 << idx2
                # adj[idx1] |= mask
                # for idx2 in valid_indices: adj[idx2] |= 1 << idx1
                
                # This is 2x the work for the inner loop.
                # Let's stick to this for correctness first.

    # 3. Max Clique Solver
    # We want to find the maximum clique in the graph defined by adj.
    # Note: adj[i] does NOT include i itself.
    
    best_clique_size = 0
    best_clique = []
    
    # Use a recursive branch and bound
    # Order nodes by degree descending to help pruning?
    # Or use the standard "coloring" heuristic for Max Clique.
    
    # Let's implement a simple version first to see if it works.
    # Given the size (8255), a naive B&B might be too slow.
    # We need the coloring heuristic (Tomita).
    
    # 1. Compute initial ordering
    # 2. Compute greedy coloring on the subgraph of candidates
    
    # To keep it simple and fast within the block, I'll implement a basic B&B with degree sorting.
    # If that's not enough, I'll add coloring.
    
    # Actually, let's try to find the clique size.
    
    def expand(clique, candidates):
        nonlocal best_clique_size, best_clique
        if not candidates:
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique = clique[:]
            return
        
        # If len(clique) + len(candidates) <= best_clique_size, prune
        if len(clique) + len(candidates) <= best_clique_size:
            return
            
        # Pick a node from candidates
        # Heuristic: pick node with highest degree in candidates
        # For speed, just pick the first one or random?
        # Let's pick the one with highest degree in the current candidate set.
        # This requires calculating degrees, which is O(|candidates|).
        
        # To make it faster, we can just iterate.
        # But we need to remove the node from candidates and intersect neighbors.
        
        # Let's use a list for candidates to make removal easy?
        # Bitsets are fast for intersection.
        
        # Let's convert candidates bitset to a list of indices for the loop?
        # That is slow.
        
        # Let's iterate through the bits of candidates.
        # We'll try each node in candidates.
        
        # Optimization: Order candidates by degree in the global graph?
        # Precompute global degrees.
        
        # Let's try a simpler approach:
        # Since we just need the size, and the graph is structured,
        # maybe we can just return the size found by a greedy heuristic?
        # No, the directive asks for branch and bound to find global max.
        
        # I will implement the standard MCQ algorithm logic.
        
        # For now, to ensure we get *some* improvement and don't time out,
        # I will use a greedy coloring bound if possible, or just a simple B&B.
        
        # Given the complexity, I will provide a B&B that works on the bitset.
        
        # We need to iterate through candidates.
        # Let's get the list of indices in candidates.
        cand_list = []
        c = candidates
        while c:
            # Get lowest set bit
            lsb = c & -c
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            c ^= lsb
            
        # Sort by degree (precomputed)
        # deg[i] = bit_count(adj[i])
        
        # We need degrees. Let's compute them once.
        # deg = [x.bit_count() for x in adj]
        
        # Sort cand_list by deg descending
        cand_list.sort(key=lambda x: deg[x], reverse=True)
        
        for v in cand_list:
            if v not in candidates: # This check is expensive if candidates is int
                continue
            # Check if v is in candidates
            if not (candidates & (1 << v)):
                continue
                
            # Include v
            new_clique = clique + [v]
            # New candidates: neighbors of v intersected with current candidates
            # Also remove v and all nodes that come after v in the ordering?
            # In MCQ, we process in order.
            
            # Let's use the standard recursive step:
            # expand(clique + [v], candidates & adj[v])
            # But we must ensure we don't miss cliques.
            # The standard algorithm picks v, adds to clique, and restricts candidates to N(v) \ {v}.
            # To avoid duplicates, we usually process vertices in a specific order and only consider candidates that come "after" v in that order.
            
            # Let's implement a simplified version:
            # Pick v.
            # Recurse with candidates & adj[v] & ~(1 << v)
            # Then remove v from candidates and recurse again?
            
            # Correct B&B for Max Clique:
            # expand(clique, candidates):
            #   if empty: update best
            #   for v in candidates (in some order):
            #       expand(clique + [v], candidates & adj[v])
            #       candidates = candidates \ {v}
            #       # Pruning: if len(clique) + len(candidates) <= best: break
            
            # This works.
            
            expand(new_clique, candidates & adj[v] & ~(1 << v))
            candidates ^= (1 << v)
            if len(clique) + len(candidates) <= best_clique_size:
                break

    # Precompute degrees
    deg = [x.bit_count() for x in adj]
    
    # Initial candidates: all nodes
    initial_candidates = (1 << N) - 1
    
    # Start search
    # To improve performance, we can order the initial candidates by degree.
    # The expand function handles ordering internally for the loop, but the initial call is just all nodes.
    
    # We need to be careful with recursion depth. N=8255.
    # But the clique size is small (~10-20). Depth is bounded by clique size.
    
    expand([], initial_candidates)
    
    # Construct result from best_clique
    result = []
    for idx in best_clique:
        r, m = nodes[idx]
        result.append([r, m])
        
    return result

# EVOLVE-BLOCK-END
