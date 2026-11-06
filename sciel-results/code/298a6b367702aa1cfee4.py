"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_pairs)}

    # ---- Precompute Q3 masks (12-edge 3-regular submasks of K4,4) ----
    def is_q3_mask(mask):
        da = [0]*4
        db = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    da[i] += 1
                    db[j] += 1
        return all(d == 3 for d in da) and all(d == 3 for d in db)

    q3_masks = []
    for combo in itertools.combinations(range(16), 12):
        m = 0
        for idx in combo:
            m |= (1 << idx)
        if is_q3_mask(m):
            q3_masks.append(m)
    q3_masks = tuple(q3_masks)  # 6 masks

    # ---- Precompute all K4,4 splits (A of size 4, B of size 4) ----
    all_verts = list(range(n))
    k44_list = []  # each: (A tuple, B tuple, pairs [(a,b,bitpos),...])
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            pairs = []
            for i in range(4):
                for j in range(4):
                    pairs.append((A[i], B[j], i*4 + j))
            k44_list.append((A, B, pairs))

    # ---- Index: for each edge, which K4,4 splits contain it ----
    edge_to_k44 = {}
    for idx, (A, B, pairs) in enumerate(k44_list):
        for a, b, _ in pairs:
            e = (a, b) if a < b else (b, a)
            edge_to_k44.setdefault(e, []).append(idx)

    # ---- Build graph as bitsets ----
    adj = [0] * n
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    def edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    # ---- Seed: K7 + K7 + K2 (57 edges, Q3-free) ----
    for i in range(7):
        for j in range(i+1, 7):
            add_edge(i, j)
    for i in range(7, 14):
        for j in range(i+1, 14):
            add_edge(i, j)
    add_edge(14, 15)

    best_edges = edge_count()
    best_adj = adj[:]

    def time_left():
        return deadline - time.time()

    # ---- Check if adding (u,v) creates a Q3 ----
    def creates_q3(u, v):
        e = (u, v) if u < v else (v, u)
        for kidx in edge_to_k44.get(e, ()):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if (a == u and b == v) or (a == v and b == u):
                    mask |= (1 << bit)  # the new edge
                elif has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    return True
        return False

    # ---- Count number of Q3 subgraphs in current graph ----
    def count_q3():
        cnt = 0
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    cnt += 1
        return cnt

    # ---- Greedy: try to add every missing edge ----
    missing = [e for e in all_pairs if not has_edge(*e)]
    random.seed(42)
    random.shuffle(missing)

    for u, v in missing:
        if time_left() < 15:
            break
        if not creates_q3(u, v):
            add_edge(u, v)
            if edge_count() > best_edges:
                best_edges = edge_count()
                best_adj = adj[:]

    # ---- Simulated Annealing on edge set ----
    if time_left() > 20:
        adj[:] = best_adj[:]
        sa_edges = edge_count()
        sa_q3 = count_q3()

        T = 5.0
        cooling = 0.9995
        min_T = 0.1
        all_pairs_list = all_pairs[:]

        while time_left() > 10 and T > min_T:
            u, v = random.choice(all_pairs_list)
            if has_edge(u, v):
                remove_edge(u, v)
                new_edges = sa_edges - 1
                new_q3 = count_q3()
                new_score = new_edges - 100 * new_q3
                old_score = sa_edges - 100 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
                new_edges = sa_edges + 1
                new_q3 = count_q3()
                new_score = new_edges - 100 * new_q3
                old_score = sa_edges - 100 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)

            T *= cooling

        if count_q3() > 0:
            for _ in range(100):
                if count_q3() == 0:
                    break
                if time_left() < 5:
                    break
                found = False
                for u, v in all_pairs:
                    if not has_edge(u, v):
                        continue
                    old_q3 = count_q3()
                    remove_edge(u, v)
                    new_q3 = count_q3()
                    if new_q3 < old_q3:
                        found = True
                        break
                    add_edge(u, v)
                if not found:
                    break

    # ---- Genetic Algorithm with repair ----
    if time_left() > 30:
        # Population of valid Q3-free graphs
        population = [best_adj[:]]
        pop_size = 8
        # Add some random valid neighbors to seed population
        for _ in range(pop_size - 1):
            cand = best_adj[:]
            # Try adding a few random edges
            miss_list = [e for e in all_pairs if not has_edge(*e)]
            random.shuffle(miss_list)
            for u, v in miss_list[:5]:
                if not creates_q3(u, v):
                    add_edge(u, v)
            population.append(cand[:])

        # GA loop
        gen = 0
        while time_left() > 15:
            gen += 1
            # Fitness: number of edges (all are valid Q3-free)
            fitness = [edge_count() for _ in range(len(population))]
            
            # Selection: tournament selection
            def select():
                a = random.randint(0, len(population)-1)
                b = random.randint(0, len(population)-1)
                if fitness[a] >= fitness[b]:
                    return population[a]
                else:
                    return population[b]
            
            # Crossover: uniform crossover on edge presence
            def crossover(p1, p2):
                child = [0]*n
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            if has_edge_in(p1, i, j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                        else:
                            if has_edge_in(p2, i, j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                return child
            
            def has_edge_in(adj_list, u, v):
                return bool(adj_list[u] & (1 << v))
            
            # Repair: remove edges that create Q3s, then greedily add back
            def repair(adj_list):
                # First, check if valid; if not, remove problematic edges
                # We need to work with a temporary adj
                temp_adj = adj_list[:]
                
                # Check for Q3s and remove edges
                changed = True
                while changed and time_left() > 10:
                    changed = False
                    for kidx in range(len(k44_list)):
                        A, B, pairs = k44_list[kidx]
                        mask = 0
                        for a, b, bit in pairs:
                            if temp_adj[a] & (1 << b):
                                mask |= (1 << bit)
                        for qm in q3_masks:
                            if (mask & qm) == qm:
                                # Remove one edge from this Q3
                                # Find an edge in the Q3 that is in temp_adj
                                for a, b, bit in pairs:
                                    if (mask & (1 << bit)) and (temp_adj[a] & (1 << b)):
                                        temp_adj[a] &= ~(1 << b)
                                        temp_adj[b] &= ~(1 << a)
                                        changed = True
                                        break
                                break
                        if changed:
                            break
                
                # Now greedily add back missing edges
                for u, v in all_pairs:
                    if not (temp_adj[u] & (1 << v)):
                        # Check if adding creates Q3 in temp_adj
                        e = (u, v) if u < v else (v, u)
                        creates = False
                        for kidx in edge_to_k44.get(e, ()):
                            A, B, pairs = k44_list[kidx]
                            mask = 0
                            for a, b, bit in pairs:
                                if (a == u and b == v) or (a == v and b == u):
                                    mask |= (1 << bit)
                                elif temp_adj[a] & (1 << b):
                                    mask |= (1 << bit)
                            for qm in q3_masks:
                                if (mask & qm) == qm:
                                    creates = True
                                    break
                            if creates:
                                break
                        if not creates:
                            temp_adj[u] |= (1 << v)
                            temp_adj[v] |= (1 << u)
                
                return temp_adj

            # Generate next generation
            next_pop = []
            # Elitism: keep best
            best_idx = max(range(len(population)), key=lambda i: fitness[i])
            next_pop.append(population[best_idx][:])
            
            while len(next_pop) < pop_size:
                p1 = select()
                p2 = select()
                child = crossover(p1, p2)
                child = repair(child)
                next_pop.append(child)
            
            population = next_pop
            
            # Update best
            for i, pop_adj in enumerate(population):
                ec = sum(bin(pop_adj[i]).count("1") for i in range(n)) // 2
                if ec > best_edges:
                    # Verify validity
                    valid = True
                    for kidx in range(len(k44_list)):
                        A, B, pairs = k44_list[kidx]
                        mask = 0
                        for a, b, bit in pairs:
                            if pop_adj[a] & (1 << b):
                                mask |= (1 << bit)
                        for qm in q3_masks:
                            if (mask & qm) == qm:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_edges = ec
                        best_adj = pop_adj[:]

    # ---- Local search: 1-1 edge swaps to gain density ----
    adj[:] = best_adj[:]
    for _ in range(2000):
        if time_left() < 10:
            break
        rem_edges = [e for e in all_pairs if has_edge(*e)]
        if not rem_edges:
            break
        ru, rv = random.choice(rem_edges)
        remove_edge(ru, rv)

        miss = [e for e in all_pairs if not has_edge(*e)]
        random.shuffle(miss)
        added = 0
        for u, v in miss:
            if added >= 2:
                break
            if not creates_q3(u, v):
                add_edge(u, v)
                added += 1

        if edge_count() > best_edges:
            best_edges = edge_count()
            best_adj = adj[:]
        else:
            for u, v in miss[:added]:
                remove_edge(u, v)
            add_edge(ru, rv)

    # Restore best state
    adj[:] = best_adj[:]

    # Final validity check
    if time_left() > 5:
        found = False
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    found = True
                    break
            if found:
                break
        if found:
            adj = [0]*n
            for i in range(7):
                for j in range(i+1, 7):
                    add_edge(i, j)
            for i in range(7, 14):
                for j in range(i+1, 14):
                    add_edge(i, j)
            add_edge(14, 15)

    return [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
# EVOLVE-BLOCK-END
