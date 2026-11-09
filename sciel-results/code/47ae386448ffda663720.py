import time
import random
from random import Random
import math

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    trees = instance_trees()
    N_HOSTS = 20
    N_TREES = 20
    
    # Pre-compute tree edge lists for faster access
    tree_edges = [trees[i] for i in range(N_TREES)]
    
    # State: state[i] is a list of k host vertices for T_{i+1}
    # edge_owner[e] = set of tree indices using edge e
    # edge_count[e] = number of times edge e is used
    
    def build_edge_owner(state):
        """Build edge ownership map from state."""
        edge_owner = {}  # edge -> set of tree indices
        for i in range(N_TREES):
            k = i + 1
            if k == 1:
                continue
            mapping = state[i]
            for u, v in tree_edges[i]:
                e1, e2 = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                edge = (e1, e2)
                if edge not in edge_owner:
                    edge_owner[edge] = set()
                edge_owner[edge].add(i)
        return edge_owner
    
    def count_conflicts(edge_owner):
        """Count total conflicts from edge ownership."""
        return sum(len(owners) - 1 for owners in edge_owner.values() if len(owners) > 1)
    
    # Initialize with greedy embedding (embed larger trees first for better packing)
    def greedy_init():
        state = [None] * N_TREES
        # Embed in order: T20, T19, ..., T1
        # But we need to embed all, so let's do a smarter greedy
        # Start with T1 (trivial), then embed others greedily avoiding conflicts
        
        # First, create a random initial state
        for i in range(N_TREES):
            k = i + 1
            if k == 1:
                state[i] = [random.randint(0, N_HOSTS - 1)]
            else:
                hosts = list(range(N_HOSTS))
                random.shuffle(hosts)
                state[i] = hosts[:k]
        
        # Now do a few passes of greedy improvement
        for _ in range(50):
            improved = False
            for i in range(N_TREES):
                k = i + 1
                if k == 1:
                    continue
                # Try to find a better embedding for tree i
                best_mapping = state[i][:]
                best_conflicts = None
                current_edge_owner = build_edge_owner(state)
                current_total = count_conflicts(current_edge_owner)
                
                # Try a few random perturbations
                for _ in range(20):
                    new_mapping = best_mapping[:]
                    # Random swap
                    a, b = random.sample(range(k), 2)
                    new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
                    
                    # Calculate conflicts for this tree only (approximate)
                    # This is approximate - we'll verify later
                    if random.random() < 0.1:  # Occasionally do full check
                        state[i] = new_mapping
                        eo = build_edge_owner(state)
                        tc = count_conflicts(eo)
                        if best_conflicts is None or tc < best_conflicts:
                            best_conflicts = tc
                            best_mapping = new_mapping[:]
                        state[i] = best_mapping
            
            if best_mapping != state[i]:
                state[i] = best_mapping
                improved = True
        
        return state
    
    # Better: use a more systematic approach
    # Embed trees one by one, choosing embeddings that minimize conflicts with already-embedded trees
    
    def systematic_init():
        state = [None] * N_TREES
        edge_used = set()  # set of (u,v) edges already used
        
        # Embed T1 first (trivial)
        state[0] = [random.randint(0, N_HOSTS - 1)]
        
        # Embed T2..T20 in order of decreasing size (T20 first, then T19, etc.)
        # Actually, let's embed in a smart order: start with trees that have more constraints
        # For now, embed in order T2, T3, ..., T20
        
        for i in range(1, N_TREES):
            k = i + 1
            if k == 1:
                continue
            
            best_mapping = None
            best_conflict_count = float('inf')
            
            # Try multiple random embeddings, pick the one with fewest conflicts
            for attempt in range(100):
                hosts = list(range(N_HOSTS))
                random.shuffle(hosts)
                mapping = hosts[:k]
                
                # Count conflicts with already-embedded trees
                conflicts = 0
                for u, v in tree_edges[i]:
                    e1, e2 = min(mapping[u], mapping[v]), max(mapping[u], mapping[v])
                    if (e1, e2) in edge_used:
                        conflicts += 1
                
                if conflicts < best_conflict_count:
                    best_conflict_count = conflicts
                    best_mapping = mapping[:]
                    if conflicts == 0:
                        break
            
            state[i] = best_mapping
            # Update edge_used
            for u, v in tree_edges[i]:
                e1, e2 = min(best_mapping[u], best_mapping[v]), max(best_mapping[u], best_mapping[v])
                edge_used.add((e1, e2))
        
        return state
    
    # Start with systematic initialization
    state = systematic_init()
    
    # Build initial edge ownership
    edge_owner = build_edge_owner(state)
    current_conflicts = count_conflicts(edge_owner)
    
    best_state = [s[:] if s is not None else None for s in state]
    best_conflicts = current_conflicts
    
    # Simulated Annealing with incremental updates
    temperature = 5.0
    min_temp = 0.01
    cooling_rate = 0.995
    
    def try_move(state, i, new_mapping, edge_owner):
        """
        Try to apply a move to tree i.
        Returns (delta_conflicts, updated_edge_owner) or None if move is invalid.
        """
        k = i + 1
        if k == 1:
            return 0, edge_owner
        
        old_mapping = state[i]
        
        # Remove old edges from edge_owner
        old_edges = []
        for u, v in tree_edges[i]:
            e1, e2 = min(old_mapping[u], old_mapping[v]), max(old_mapping[u], old_mapping[v])
            old_edges.append((e1, e2))
        
        for edge in old_edges:
            if edge in edge_owner:
                edge_owner[edge].discard(i)
                if not edge_owner[edge]:
                    del edge_owner[edge]
        
        # Add new edges to edge_owner
        new_edges = []
        for u, v in tree_edges[i]:
            e1, e2 = min(new_mapping[u], new_mapping[v]), max(new_mapping[u], new_mapping[v])
            new_edges.append((e1, e2))
        
        for edge in new_edges:
            if edge not in edge_owner:
                edge_owner[edge] = set()
            edge_owner[edge].add(i)
        
        # Calculate delta in conflicts
        # For each edge, conflict count is max(0, len(owners) - 1)
        # We need to compute the change
        
        # Actually, it's easier to just recompute total conflicts for affected edges
        # But that's still O(edges). Let's just recompute total.
        new_conflicts = count_conflicts(edge_owner)
        delta = new_conflicts - current_conflicts
        
        return delta, edge_owner
    
    # Main SA loop
    iteration = 0
    while time.time() < deadline:
        if current_conflicts == 0:
            break
        
        # Choose a tree to modify
        # Weight towards trees that are involved in more conflicts
        i = random.randint(0, N_TREES - 1)
        k = i + 1
        if k == 1:
            continue
        
        old_mapping = state[i][:]
        
        # Generate new mapping
        r = random.random()
        
        if r < 0.4:
            # Swap two vertices
            new_mapping = old_mapping[:]
            a, b = random.sample(range(k), 2)
            new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif r < 0.7:
            # Move one vertex to a random available host
            new_mapping = old_mapping[:]
            a = random.randint(0, k - 1)
            used_hosts = set(new_mapping)
            available = [h for h in range(N_HOSTS) if h not in used_hosts]
            if available:
                new_mapping[a] = random.choice(available)
            else:
                # No available, do a swap
                b = random.randint(0, k - 1)
                new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
        elif r < 0.9:
            # Partial random restart: keep some vertices, randomize others
            new_mapping = old_mapping[:]
            num_to_randomize = max(2, k // 3)
            positions = random.sample(range(k), num_to_randomize)
            for pos in positions:
                used_hosts = set(new_mapping)
                available = [h for h in range(N_HOSTS) if h not in used_hosts]
                if available:
                    new_mapping[pos] = random.choice(available)
        else:
            # Full random restart
            hosts = list(range(N_HOSTS))
            random.shuffle(hosts)
            new_mapping = hosts[:k]
        
        # Apply move and calculate delta
        # We need to be careful with edge_owner updates
        # Let's use a simpler approach: save state, apply, compute, revert if needed
        
        # Save old edge_owner state (shallow copy of the sets we'll modify)
        # Actually, let's just recompute conflicts from scratch - it's fast enough
        # 20 trees * ~10 edges avg = 200 edge lookups per conflict calc
        
        state[i] = new_mapping
        new_edge_owner = build_edge_owner(state)
        new_conflicts = count_conflicts(new_edge_owner)
        delta = new_conflicts - current_conflicts
        
        # Metropolis criterion
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_conflicts = new_conflicts
            edge_owner = new_edge_owner
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_state = [s[:] if s is not None else None for s in state]
        else:
            state[i] = old_mapping
            # edge_owner stays as is (we didn't modify it in place)
        
        # Cool down
        temperature = max(min_temp, temperature * cooling_rate)
        iteration += 1
    
    return best_state
