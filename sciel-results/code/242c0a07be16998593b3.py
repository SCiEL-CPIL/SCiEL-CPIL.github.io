import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Target: Minimize distinct prime factors of all pairwise sums.
    # Strategy: Choose a_i = K * x_i.
    # Sums are K * (x_i + x_j).
    # Prime factors = Primes(K) U Primes(x_i + x_j).
    # We want Primes(x_i + x_j) to be a subset of Primes(K) or a very small set.
    # Ideally, Primes(x_i + x_j) subset of Primes(K).
    # Then total primes = Primes(K).
    
    # Try K = 210 (2*3*5*7). Primes(K) = {2, 3, 5, 7}. Count = 4.
    # Need 12 distinct x_i <= 10000/210 = 47.
    # Such that x_i + x_j is 7-smooth (primes in {2,3,5,7}).
    
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    def find_clique(n_nodes, adj, target_size, time_limit):
        # adj[i] is a set of nodes connected to i
        # We want a clique of size target_size.
        # Simple backtracking.
        best = []
        
        def backtrack(current_clique, candidates):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(current_clique) == target_size:
                best = current_clique[:]
                return
            # If not enough candidates, prune
            if len(candidates) < target_size - len(current_clique):
                return
            
            # Heuristic: Pick candidate with highest degree in remaining graph?
            # Or just iterate.
            for i in range(len(candidates)):
                node = candidates[i]
                # Check if node connects to all in current_clique
                # Actually, we maintain candidates as nodes connected to ALL in current_clique.
                # So we just pick one.
                
                new_clique = current_clique + [node]
                # New candidates: intersection of adj[node] and remaining candidates
                new_candidates = [c for c in candidates[i+1:] if c in adj[node]]
                
                backtrack(new_clique, new_candidates)
                if best:
                    return

        # Initial candidates: all nodes
        # To improve speed, we can sort nodes by degree
        nodes = list(range(n_nodes))
        # adj is 0-indexed in this function, but our x_i are 1-indexed.
        # Let's map x_i (1..47) to index (0..46).
        backtrack([], nodes)
        return best

    # 1. Attempt with K=210, Primes={2,3,5,7}
    K = 210
    max_x = 10000 // K
    primes_K = [2, 3, 5, 7]
    
    # Precompute valid sums
    valid_sums = set()
    for s in range(2, 2 * max_x + 1):
        if is_smooth(s, primes_K):
            valid_sums.add(s)
            
    # Build graph
    n_nodes = max_x
    adj = [set() for _ in range(n_nodes)]
    for u in range(1, max_x + 1):
        for v in range(u + 1, max_x + 1):
            if u + v in valid_sums:
                adj[u-1].add(v-1)
                adj[v-1].add(u-1)
                
    # Find clique of size 12
    # We need 12 distinct x_i.
    target_size = 12
    result_idx = find_clique(n_nodes, adj, target_size, deadline)
    
    if result_idx:
        # result_idx contains 0-indexed nodes. Convert to 1-indexed x_i.
        x_values = [i + 1 for i in result_idx]
        # Verify
        valid = True
        for i in range(len(x_values)):
            for j in range(i+1, len(x_values)):
                if not is_smooth(x_values[i] + x_values[j], primes_K):
                    valid = False
                    break
            if not valid: break
        
        if valid:
            a_values = [K * x for x in x_values]
            # Check constraints
            if len(set(a_values)) == 12 and max(a_values) <= 10000:
                return a_values
        else:
            # Fallback if verification fails (shouldn't happen)
            pass

    # 2. Attempt with K=210, but allow one more prime in sums?
    # If we allow 11 in sums, total primes = {2,3,5,7,11}. Count = 5.
    # This relaxes the constraint on x_i + x_j.
    primes_allowed_in_sums = [2, 3, 5, 7, 11]
    
    valid_sums_5 = set()
    for s in range(2, 2 * max_x + 1):
        if is_smooth(s, primes_allowed_in_sums):
            valid_sums_5.add(s)
            
    adj_5 = [set() for _ in range(n_nodes)]
    for u in range(1, max_x + 1):
        for v in range(u + 1, max_x + 1):
            if u + v in valid_sums_5:
                adj_5[u-1].add(v-1)
                adj_5[v-1].add(u-1)
                
    result_idx_5 = find_clique(n_nodes, adj_5, target_size, deadline)
    
    if result_idx_5:
        x_values = [i + 1 for i in result_idx_5]
        valid = True
        for i in range(len(x_values)):
            for j in range(i+1, len(x_values)):
                if not is_smooth(x_values[i] + x_values[j], primes_allowed_in_sums):
                    valid = False
                    break
            if not valid: break
            
        if valid:
            a_values = [K * x for x in x_values]
            if len(set(a_values)) == 12 and max(a_values) <= 10000:
                return a_values

    # 3. Fallback: Use a smaller K to allow more x_i, but same prime set for K.
    # Try K=30 (2*3*5). Primes(K)={2,3,5}.
    # Need x_i + x_j to be 5-smooth.
    # Max x_i = 10000/30 = 333.
    # Search space is larger, but graph is sparser.
    # Given time limit, this might be risky, but let's try a small subset or heuristic.
    # Actually, if 4 primes wasn't found, 5 primes with K=210 is a good bet.
    # If that failed, maybe try K=60 (2*2*3*5)? Primes {2,3,5}.
    # Sums 60(x_i+x_j). Primes {2,3,5} U Primes(x_i+x_j).
    # If x_i+x_j is 5-smooth, total primes {2,3,5}. Count 3.
    # This is a better target than 4!
    
    # Let's prioritize K=60 or K=30 to get 3 primes.
    # K=60: max_x = 166.
    # Primes in K: {2, 3, 5}.
    # We want x_i + x_j to be 5-smooth.
    
    K3 = 60
    max_x3 = 10000 // K3
    primes_K3 = [2, 3, 5]
    
    valid_sums_3 = set()
    for s in range(2, 2 * max_x3 + 1):
        if is_smooth(s, primes_K3):
            valid_sums_3.add(s)
            
    n_nodes3 = max_x3
    adj_3 = [set() for _ in range(n_nodes3)]
    for u in range(1, max_x3 + 1):
        for v in range(u + 1, max_x3 + 1):
            if u + v in valid_sums_3:
                adj_3[u-1].add(v-1)
                adj_3[v-1].add(u-1)
                
    # Re-use find_clique. It might be slower due to 166 nodes.
    # But if we find it, score is 3.
    
    # Optimization: Sort nodes by degree before backtracking to find large cliques faster.
    # My find_clique doesn't sort. Let's modify the call or the function.
    # For now, let's just try. If it times out, we return the 4-prime solution if found, or a heuristic.
    
    # Actually, the previous find_clique might be too slow for 166 nodes if the graph is sparse.
    # Let's check if we have time.
    if time.time() < deadline - 5.0:
         # We can try a smarter clique finder or just trust the 4-prime solution.
         # Given the "minimize" goal, 3 is better than 4.
         # But 4 is safer to find.
         # Let's try to find a 3-prime solution.
         
         # Simple heuristic for 3-prime:
         # Pick x_i from small numbers where sums are very smooth.
         # Small sums are more likely to be smooth.
         # Try x_i in 1..20.
         subset_x = list(range(1, 21))
         # Filter subset_x to keep only those that form a clique?
         # No, just pick 12 from a dense region.
         
         # Let's just return the best found so far.
         # If result_idx (4 primes) was found, we should have returned it.
         # If not, we try to find one quickly.
         
         # Let's try a greedy approach for 3 primes (K=60)
         # Pick x_1=1.
         # x_2 must satisfy 1+x_2 is 5-smooth.
         # ...
         
         # This is getting complex. Let's stick to the 4-prime solution (K=210) as primary.
         # If that fails, return a valid set with a known prime count (e.g. 5 or 6).
         
         # Fallback: Return a set that definitely works.
         # a_i = 210 * i for i in 1..12.
         # Sums 210(i+j).
         # i+j ranges 3..23.
         # Primes in i+j: 2,3,5,7,11,13,17,19,23.
         # Total primes: {2,3,5,7,11,13,17,19,23}. Count 9.
         
         # Can we do better with a simple pattern?
         # a_i = 2 * 3 * 5 * 7 * i = 210 i.
         # Same as above.
         
         # Try a_i = 2^i? No, distinct.
         
         # Let's just return the 4-prime solution if found.
         # If not found, we'll return a set constructed to have 5 primes.
         # K=210, allow 11 in sums.
         # We already tried this (result_idx_5).
         
    # If we reach here, we didn't find a 4-prime or 5-prime clique.
    # Return a valid set with a reasonable score.
    # Let's try to construct a set with 5 primes manually.
    # Primes: 2, 3, 5, 7, 11.
    # Use K=210.
    # Pick x_i such that x_i+x_j is 11-smooth.
    # We did this search. If it failed, maybe the clique size 12 is hard.
    
    # Let's try a different K.
    # K = 2 * 3 * 5 * 7 * 11 = 2310.
    # Primes(K) = {2,3,5,7,11}. Count 5.
    # Max x_i = 10000/2310 = 4.
    # Not enough distinct x_i (need 12).
    
    # So K must be smaller.
    # K=210 is the best for 4 primes.
    
    # If search failed, maybe my clique finder is too weak.
    # Let's try a random restart or a simpler greedy for the 4-prime case.
    
    # Greedy for 4-prime (K=210):
    # 1. Start with x=1.
    # 2. Add x=2 if 1+2=3 is smooth. Yes.
    # 3. Add x=3 if 1+3=4, 2+3=5 are smooth. Yes.
    # 4. Add x=4 if 1+4=5, 2+4=6, 3+4=7 are smooth. Yes.
    # 5. Add x=5 if 1+5=6, 2+5=7, 3+5=8, 4+5=9 are smooth. Yes.
    # 6. Add x=6 if 1+6=7, 2+6=8, 3+6=9, 4+6=10, 5+6=11. 11 is NOT 7-smooth. Fail.
    # So {1,2,3,4,5} is a clique. Size 5.
    # Try x=7.
    # 1+7=8 (ok), 2+7=9 (ok), 3+7=10 (ok), 4+7=11 (Fail).
    # Try x=8.
    # 1+8=9 (ok), 2+8=10 (ok), 3+8=11 (Fail).
    # Try x=9.
    # 1+9=10 (ok), 2+9=11 (Fail).
    # Try x=10.
    # 1+10=11 (Fail).
    # Try x=11.
    # 1+11=12 (ok), 2+11=13 (Fail).
    # Try x=12.
    # 1+12=13 (Fail).
    # Try x=13.
    # 1+13=14 (ok), 2+13=15 (ok), 3+13=16 (ok), 4+13=17 (Fail).
    # Try x=14.
    # 1+14=15 (ok), 2+14=16 (ok), 3+14=17 (Fail).
    # Try x=15.
    # 1+15=16 (ok), 2+15=17 (Fail).
    # Try x=16.
    # 1+16=17 (Fail).
    # Try x=17.
    # 1+17=18 (ok), 2+17=19 (Fail).
    # Try x=18.
    # 1+18=19 (Fail).
    # Try x=19.
    # 1+19=20 (ok), 2+19=21 (ok), 3+19=22 (Fail).
    # Try x=20.
    # 1+20=21 (ok), 2+20=22 (Fail).
    # Try x=21.
    # 1+21=22 (Fail).
    # Try x=22.
    # 1+22=23 (Fail).
    # Try x=23.
    # 1+23=24 (ok), 2+23=25 (ok), 3+23=26 (Fail).
    # Try x=24.
    # 1+24=25 (ok), 2+24=26 (Fail).
    # Try x=25.
    # 1+25=26 (Fail).
    # Try x=26.
    # 1+26=27 (ok), 2+26=28 (ok), 3+26=29 (Fail).
    # Try x=27.
    # 1+27=28 (ok), 2+27=29 (Fail).
    # Try x=28.
    # 1+28=29 (Fail).
    # Try x=29.
    # 1+29=30 (ok), 2+29=31 (Fail).
    # Try x=30.
    # 1+30=31 (Fail).
    # Try x=31.
    # 1+31=32 (ok), 2+31=33 (Fail).
    # Try x=32.
    # 1+32=33 (Fail).
    # Try x=33.
    # 1+33=34 (Fail).
    # Try x=34.
    # 1+34=35 (ok), 2+34=36 (ok), 3+34=37 (Fail).
    # Try x=35.
    # 1+35=36 (ok), 2+35=37 (Fail).
    # Try x=36.
    # 1+36=37 (Fail).
    # Try x=37.
    # 1+37=38 (Fail).
    # Try x=38.
    # 1+38=39 (Fail).
    # Try x=39.
    # 1+39=40 (ok), 2+39=41 (Fail).
    # Try x=40.
    # 1+40=41 (Fail).
    # Try x=41.
    # 1+41=42 (ok), 2+41=43 (Fail).
    # Try x=42.
    # 1+42=43 (Fail).
    # Try x=43.
    # 1+43=44 (Fail).
    # Try x=44.
    # 1+44=45 (ok), 2+44=46 (Fail).
    # Try x=45.
    # 1+45=46 (Fail).
    # Try x=46.
    # 1+46=47 (Fail).
    # Try x=47.
    # 1+47=48 (ok), 2+47=49 (ok), 3+47=50 (ok), 4+47=51 (Fail).
    
    # It seems hard to get a large clique with just {1,2,3,4,5} as core.
    # My backtracking should find better ones if they exist.
    # If it didn't, maybe there isn't a clique of size 12.
    
    # In that case, we fall back to 5 primes.
    # If 5 primes also fails, we return a heuristic set.
    
    # Let's ensure we return something valid.
    # a_i = 210 * i for i=1..12.
    return list(range(210, 210*13, 210))

# EVOLVE-BLOCK-END
