"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    N = 42
    start_time = time.time()
    
    # We will try to build a graph on N vertices
    # adj[i] is a set of neighbors of i
    # We build vertex by vertex
    
    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() - start_time > 280:
            break
            
        adj = [set() for _ in range(N)]
        success = True
        
        for v in range(N):
            if v == 0:
                continue
                
            # Find all 4-cliques and 4-independent sets in the subgraph 0..v-1
            # This is expensive if done naively every time.
            # Optimization: We only need to check constraints for the new vertex.
            # A 4-clique in 0..v-1 is a set C such that all pairs in C are edges.
            # An independent 4-set is a set I such that no pairs in I are edges.
            
            # To speed up, we can just check the constraints on the fly during repair.
            # Start with a random assignment for edges from v to 0..v-1
            edges_to_prev = {u: (random.random() < 0.5) for u in range(v)}
            
            # Repair loop
            repaired = False
            for _ in range(1000): # Max repair steps
                violations = []
                
                # Check 4-cliques
                # If there is a 4-clique C in 0..v-1, and v is connected to all in C, violation.
                # We need to find 4-cliques.
                # For N=42, finding all 4-cliques in a 41-vertex graph is feasible but heavy if repeated 42 times.
                # Total 4-cliques in random graph p=0.5 is ~ C(41,4)/16 ~ 100k.
                # Checking each against v's edges is O(4).
                # 400k ops per repair step. 1000 steps = 400M ops. Too slow.
                
                # Better: Precompute 4-cliques and 4-indep sets once per vertex addition?
                # Still heavy.
                
                # Alternative: Use a simpler heuristic.
                # Just pick edges randomly.
                # Then check if any 5-clique or 5-indep is formed.
                # This is what the seed did, but it failed.
                
                # Let's try a different heuristic:
                # Keep the graph as "balanced" as possible.
                # For each u in 0..v-1, set edge(v,u) based on degree balance?
                
                # Actually, let's just use the "Repair" logic but optimize the violation check.
                # Instead of finding ALL 4-cliques, we can check:
                # For any subset of 4 neighbors of v, if they form a K4, it's a violation.
                # So, we need to ensure that the subgraph induced by N(v) has no K4.
                # And the subgraph induced by Non-N(v) has no Indep 4 (which means K4 in complement).
                
                # So, let G_v be the graph on 0..v-1.
                # Let S = {u | edge(v,u) is True}.
                # We require: G_v[S] has no K4.
                # Let T = {u | edge(v,u) is False}.
                # We require: G_v[T] has no K4 (in complement, so no Indep 4 in G_v).
                
                # This is a strong constraint.
                # We can try to find S such that these hold.
                # Start with random S.
                # Check if G_v[S] has K4. If yes, move a vertex from S to T (break the edge).
                # Check if G_v[T] has K4 (in complement). If yes, move a vertex from T to S (add the edge).
                
                # How to check "G_v[S] has K4" fast?
                # We can precompute all K4s in G_v.
                # Then check if any K4 is subset of S.
                # Similarly for complement.
                
                # Precomputation cost:
                # Finding all K4s in a graph of size v.
                # For v=41, this is ~100k K4s.
                # Doing this for each v=1..41 is sum of C(v,4) ~ 41^4/24 ~ 100k * 41 / 24 ~ 1.7M K4s total to find?
                # No, we recompute for each v.
                # Total work ~ sum_{v=1}^{41} (Cost to find K4s in v-vertices).
                # Cost to find K4s is O(v^4) or less with backtracking.
                # This might be too slow in Python.
                
                # Let's try a simpler approach:
                # Just use Simulated Annealing on the whole 42-vertex graph.
                # It's more robust to implement correctly.
                
                pass # Placeholder for logic inside restart loop
                
            if not success:
                continue
                
            # If we get here, we built the graph? No, the logic above was 'pass'.
            # I will switch to the Simulated Annealing approach in the actual code.
            
        # If loop finishes without break, we have a graph?
        # The 'pass' means we didn't build it.
        
    # Fallback: Return Paley 37 if search fails?
    # No, we want to improve.
    
    # Let's implement the SA approach properly.
    
    return _sa_construct(N, time_limit=280)

def _sa_construct(N, time_limit):
    start_time = time.time()
    
    # Initialize random graph
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < 0.5:
                adj[i][j] = True
                adj[j][i] = True
                
    def count_violations(adj):
        # Count K5 and Indep 5
        v_count = 0
        # K5
        for combo in combinations(range(N), 5):
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    if not adj[combo[i]][combo[j]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                v_count += 1
                
        # Indep 5
        for combo in combinations(range(N), 5):
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                v_count += 1
                
        return v_count

    current_energy = count_violations(adj)
    best_energy = current_energy
    best_adj = [row[:] for row in adj]
    
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        i, j = min(i,j), max(i,j)
        
        # Flip
        old_val = adj[i][j]
        adj[i][j] = not old_val
        adj[j][i] = not old_val
        
        new_energy = count_violations(adj)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_adj = [row[:] for row in adj]
                if best_energy == 0:
                    break
        else:
            # Revert
            adj[i][j] = old_val
            adj[j][i] = old_val
            
        T *= cooling
        
    # Return best found
    if best_energy == 0:
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if best_adj[i][j]:
                    edges.append((i, j))
        return N, edges
    else:
        # Fallback to Paley 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

# EVOLVE-BLOCK-END
