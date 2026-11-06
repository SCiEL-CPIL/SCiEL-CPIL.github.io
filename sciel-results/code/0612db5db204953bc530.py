import time
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(all_pairs)
    
    # Precompute Q3 structure
    # Q3 is bipartite: even parity {0,3,5,6} and odd parity {1,2,4,7}
    # In 3-bit representation:
    # 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Even parity (sum of bits even): 0(000),3(011),5(101),6(110)
    # Odd parity (sum of bits odd): 1(001),2(010),4(100),7(111)
    even_bits = [0, 3, 5, 6]  # indices into the 8-vertex subset
    odd_bits = [1, 2, 4, 7]
    
    # Q3 edges: connect v to v^1, v^2, v^4 (differ in one bit)
    q3_edges = []
    for v in even_bits:
        for bit in [1, 2, 4]:
            neighbor = v ^ bit
            if neighbor in odd_bits:
                q3_edges.append((v, neighbor))
    q3_edges_set = set(q3_edges)
    
    # Precompute all 8-vertex subsets and their Q3 embeddings
    subsets_8 = list(combinations(range(n), 8))
    
    # For each 8-subset, precompute all possible Q3 embeddings
    # An embedding is a mapping from Q3 vertices {0..7} to the 8 vertices
    # such that Q3 edges map to edges in the graph
    # We represent embeddings as (even_perm, odd_perm) pairs
    
    perms_4 = list(permutations(range(4)))
    
    # Precompute valid (even_perm, odd_perm) pairs for Q3
    # even_perm[i] = which of the 4 even positions maps to Q3 vertex even_bits[i]
    # odd_perm[j] = which of the 4 odd positions maps to Q3 vertex odd_bits[j]
    valid_embeddings = []
    for pe in perms_4:
        for po in perms_4:
            # Check if this embedding preserves Q3 edges
            # For each Q3 edge (e, o) where e in even_bits, o in odd_bits:
            # The mapped vertices are even_positions[pe_inv[e_idx]] and odd_positions[po_inv[o_idx]]
            # Actually, let's think differently:
            # pe maps position index -> Q3 vertex index among even
            # So if we have 8 vertices s[0..7], we assign:
            #   s[even_bits[i]] gets Q3 vertex even_bits[pe[i]]
            #   s[odd_bits[j]] gets Q3 vertex odd_bits[po[j]]
            # Then Q3 edge (a,b) exists if there's an edge between s[even_bits[i]] and s[odd_bits[j]]
            # where even_bits[pe[i]] and odd_bits[po[j]] are adjacent in Q3
            
            # Build mapping from Q3 vertex to position in subset
            q3_to_pos = {}
            for i in range(4):
                q3_to_pos[even_bits[pe[i]]] = even_bits[i]
            for j in range(4):
                q3_to_pos[odd_bits[po[j]]] = odd_bits[j]
            
            # Check all Q3 edges
            ok = True
            for a, b in q3_edges_set:
                pa = q3_to_pos[a]
                pb = q3_to_pos[b]
                # These positions must be connected in the bipartite sense
                # We'll check this against the actual graph later
                pass
            
            # Store the embedding as (pe, po)
            valid_embeddings.append((pe, po))
    
    # Actually, all (pe, po) pairs are valid embeddings of Q3 structure
    # The question is whether the graph has the required edges
    num_embeddings = len(valid_embeddings)
    
    def has_q3(adj, edge_set=None):
        """
        adj: 16x16 numpy array of bools
        Returns True if the graph contains Q3 as a subgraph.
        Uses precomputed subsets and embeddings.
        """
        for s in subsets_8:
            # Quick check: count edges within this subset
            ec = 0
            for i in range(8):
                vi = s[i]
                for j in range(i+1, 8):
                    if adj[vi, s[j]]:
                        ec += 1
                        if ec >= 12:
                            break
                if ec >= 12:
                    break
            if ec < 12:
                continue
            
            # Build 4x4 bipartite adjacency between even and odd positions
            # even_pos[i] = s[even_bits[i]], odd_pos[j] = s[odd_bits[j]]
            bipartite = np.zeros((4, 4), dtype=bool)
            for i in range(4):
                for j in range(4):
                    bipartite[i, j] = adj[s[even_bits[i]], s[odd_bits[j]]]
            
            # Try all embeddings
            for pe, po in valid_embeddings:
                # Check if Q3 edges exist under this embedding
                ok = True
                for a, b in q3_edges_set:
                    # Find positions of a and b
                    if a in even_bits:
                        ia = even_bits.index(a)
                        # pe[ia] gives the Q3 even vertex at this position
                        # We need to find which position in s has Q3 vertex a
                        # q3_to_pos: Q3 vertex -> position index
                        # Position of Q3 vertex a among evens:
                        # pe[i] = Q3 even vertex at position i
                        # So position of Q3 vertex a is i where pe[i] == a
                        # Let's precompute inverse
                        pass
                
                # Simpler approach: for each Q3 edge (a,b), check if the corresponding
                # pair of positions in the subset are connected
                # For embedding (pe, po):
                #   Position even_bits[i] in subset maps to Q3 vertex even_bits[pe[i]]
                #   Position odd_bits[j] in subset maps to Q3 vertex odd_bits[po[j]]
                # So Q3 edge (even_bits[x], odd_bits[y]) corresponds to:
                #   Find i such that pe[i] = x, find j such that po[j] = y
                #   Check if bipartite[i, j] is True
                
                # Precompute inverse permutations
                pe_inv = [0]*4
                po_inv = [0]*4
                for i in range(4):
                    pe_inv[pe[i]] = i
                for j in range(4):
                    po_inv[po[j]] = j
                
                ok = True
                for a, b in q3_edges_set:
                    # a is an even Q3 vertex, b is an odd Q3 vertex
                    xa = even_bits.index(a)  # index among even Q3 vertices
                    xb = odd_bits.index(b)   # index among odd Q3 vertices
                    i = pe_inv[xa]  # position in subset for Q3 vertex a
                    j = po_inv[xb]  # position in subset for Q3 vertex b
                    if not bipartite[i, j]:
                        ok = False
                        break
                if ok:
                    return True
        return False
    
    # Initialize with a random dense graph
    random.seed(42)
    
    def make_adj(edge_set):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edge_set:
            adj[u, v] = True
            adj[v, u] = True
        return adj
    
    # Start with complete graph and remove edges to make Q3-free
    current_edges = set(all_pairs)
    adj = make_adj(current_edges)
    
    # Remove Q3 subgraphs
    removed_count = 0
    while time.time() < deadline - 30 and has_q3(adj):
        # Find a Q3 subgraph and remove one of its edges
        # For efficiency, just remove a random edge and check
        edge_list = list(current_edges)
        if not edge_list:
            break
        rem_u, rem_v = random.choice(edge_list)
        adj[rem_u, rem_v] = False
        adj[rem_v, rem_u] = False
        current_edges.remove((rem_u, rem_v))
        removed_count += 1
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Genetic algorithm
    population_size = 20
    population = [set(best_edges)]
    
    # Add some random variations
    for _ in range(population_size - 1):
        if time.time() > deadline - 20:
            break
        pop = set(best_edges)
        # Randomly add/remove some edges
        non_edges = [e for e in all_pairs if e not in pop]
        for _ in range(random.randint(5, 15)):
            if non_edges:
                e = random.choice(non_edges)
                if random.random() < 0.5:
                    pop.add(e)
                    non_edges.remove(e)
                else:
                    # Remove a random edge
                    if pop:
                        e2 = random.choice(list(pop))
                        pop.remove(e2)
        # Repair: remove edges to make Q3-free
        adj_pop = make_adj(pop)
        while has_q3(adj_pop):
            edge_list = list(pop)
            if not edge_list:
                break
            rem_u, rem_v = random.choice(edge_list)
            adj_pop[rem_u, rem_v] = False
            adj_pop[rem_v, rem_u] = False
            pop.remove((rem_u, rem_v))
        population.append(pop)
    
    # Evolution loop
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Select parents (tournament selection)
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, min(k, len(pop)))
            return max(candidates, key=lambda x: len(x))
        
        # Create offspring
        new_population = []
        for _ in range(population_size):
            if time.time() > deadline - 5:
                break
            
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover: combine edge sets
            child = set(parent1)
            # Add some edges from parent2 that aren't in parent1
            extra_edges = [e for e in parent2 if e not in child]
            random.shuffle(extra_edges)
            for e in extra_edges[:random.randint(0, 10)]:
                child.add(e)
            
            # Mutation: random add/remove
            non_edges = [e for e in all_pairs if e not in child]
            for _ in range(random.randint(3, 8)):
                if random.random() < 0.5 and non_edges:
                    e = random.choice(non_edges)
                    child.add(e)
                    non_edges.remove(e)
                elif child and random.random() < 0.3:
                    e = random.choice(list(child))
                    child.remove(e)
            
            # Repair: remove edges to make Q3-free
            adj_child = make_adj(child)
            while has_q3(adj_child):
                edge_list = list(child)
                if not edge_list:
                    break
                rem_u, rem_v = random.choice(edge_list)
                adj_child[rem_u, rem_v] = False
                adj_child[rem_v, rem_u] = False
                child.remove((rem_u, rem_v))
            
            new_population.append(child)
        
        # Update best
        for pop in new_population:
            if len(pop) > best_count:
                best_count = len(pop)
                best_edges = set(pop)
        
        # Replace population (elitism: keep best)
        population = new_population
        population.append(best_edges)
    
    # Final local search on best solution
    if time.time() < deadline - 3:
        adj = make_adj(best_edges)
        non_edges = [e for e in all_pairs if e not in best_edges]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 1:
                break
            if (u, v) in best_edges:
                continue
            adj[u, v] = True
            adj[v, u] = True
            best_edges.add((u, v))
            
            if has_q3(adj):
                adj[u, v] = False
                adj[v, u] = False
                best_edges.remove((u, v))
            else:
                best_count = len(best_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
