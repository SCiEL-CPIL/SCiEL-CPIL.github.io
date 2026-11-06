import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        
        # Common neighbors of u and v must be <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # Fast pre-check: count how many w in N(u) already have >=2 common with v
        # and how many w in N(v) already have >=2 common with u
        # If either count >= 1, adding (u,v) would create a violation
        # Actually we need: for each w in N(u), common(v,w) <= 1 after adding (u,v)
        # After adding, common(v,w) includes u if u in N(w), which it is since w in N(u)
        # So common(v,w) after = (adj[v] & adj[w]) | (1<<u if u in adj[w] else 0)
        # Since w in N(u), u is in adj[w], so we add 1 to common(v,w)
        # So we need (adj[v] & adj[w]).bit_count() <= 1 for all w in N(u)
        # Similarly (adj[u] & adj[w]).bit_count() <= 1 for all w in N(v)
        
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if (adj[v] & adj[w]).bit_count() > 1:
                return False
            w_mask ^= lsb
        
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if (adj[u] & adj[w]).bit_count() > 1:
                return False
            w_mask ^= lsb
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count

    def remove_k23_violations(adj, count):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    cnt = common.bit_count()
                    if cnt > 2:
                        common_list = []
                        mask = common
                        while mask:
                            lsb = mask & -mask
                            w = lsb.bit_length() - 1
                            common_list.append(w)
                            mask ^= lsb
                        
                        w = random.choice(common_list)
                        
                        if random.random() < 0.5:
                            remove_edge(adj, u, w)
                            count -= 1
                        else:
                            remove_edge(adj, v, w)
                            count -= 1
                        changed = True
        return count

    def crossover(parent1, parent2):
        adj = [0] * n
        count = 0
        
        for i in range(n):
            adj[i] = parent1[i] | parent2[i]
        
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    count += 1
                mask ^= lsb
        
        count = remove_k23_violations(adj, count)
        count = greedy_fill(adj, count)
        
        return adj, count

    def simulated_annealing(adj, count):
        if time.time() > deadline - 5:
            return count
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        while T > min_T and time.time() < deadline - 2:
            move_type = random.random()
            
            if move_type < 0.4:
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
            elif move_type < 0.8:
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    count = greedy_fill(adj, count)
            else:
                present = get_present_edges(adj)
                missing = get_missing_edges(adj)
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    remove_edge(adj, u1, v1)
                    count -= 1
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        count += 1
                    else:
                        add_edge(adj, u1, v1)
                        count += 1
            
            T *= cooling_rate
        
        return count

    def final_polish(adj, count):
        """Exhaustive local search: try all single-edge swaps and double-edge swaps."""
        if time.time() > deadline - 3:
            return count
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # Try adding any missing edge
            for u, v in all_edges:
                if not (adj[u] >> v) & 1:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try single-edge swap: remove (a,b), add (c,d)
            present = get_present_edges(adj)
            missing = get_missing_edges(adj)
            random.shuffle(present)
            random.shuffle(missing)
            
            for a, b in present:
                if time.time() > deadline - 1:
                    break
                remove_edge(adj, a, b)
                count -= 1
                
                best_gain = 0
                best_add = None
                for c, d in missing:
                    if is_valid_add(adj, c, d):
                        if best_add is None:
                            best_gain = 1
                            best_add = (c, d)
                
                if best_add:
                    add_edge(adj, best_add[0], best_add[1])
                    count += 1
                    improved = True
                else:
                    add_edge(adj, a, b)
                    count += 1
            
        return count

    # Population-based genetic algorithm
    pop_size = 8
    population = []
    
    # Initialize population with random greedy fills
    for _ in range(pop_size):
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        count = greedy_fill(adj, count)
        population.append((adj, count))
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Genetic algorithm loop
    gen = 0
    while time.time() < deadline - 10:
        gen += 1
        
        population.sort(key=lambda x: x[1], reverse=True)
        
        new_pop = [population[0], population[1]]
        
        while len(new_pop) < pop_size and time.time() < deadline - 10:
            idx1 = random.randint(0, min(3, len(population) - 1))
            idx2 = random.randint(0, min(3, len(population) - 1))
            if population[idx1][1] >= population[idx2][1]:
                parent1 = population[idx1][0]
            else:
                parent1 = population[idx2][0]
            
            idx3 = random.randint(0, min(3, len(population) - 1))
            idx4 = random.randint(0, min(3, len(population) - 1))
            if population[idx3][1] >= population[idx4][1]:
                parent2 = population[idx3][0]
            else:
                parent2 = population[idx4][0]
            
            child_adj, child_count = crossover(parent1, parent2)
            
            if random.random() < 0.3:
                present = get_present_edges(child_adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(child_adj, u, v)
                    child_count -= 1
                    child_count = greedy_fill(child_adj, child_count)
            
            new_pop.append((child_adj, child_count))
        
        population = new_pop
        
        for adj, count in population:
            if count > best_count:
                best_count = count
                best_adj = adj[:]
        
        best_adj_sa, best_count_sa = population[0]
        best_count_sa = simulated_annealing(best_adj_sa, best_count_sa)
        if best_count_sa > best_count:
            best_count = best_count_sa
            best_adj = best_adj_sa[:]
            population[0] = (best_adj_sa, best_count_sa)
    
    # Final polish pass on the best solution
    if time.time() < deadline - 2:
        final_count = final_polish(best_adj, best_count)
        if final_count > best_count:
            best_count = final_count
    
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges

if __name__ == "__main__":
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
    n = 20
    adj = [0] * n
    for u, v in res:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                valid = False
                print(f"Invalid: {u}, {v} have {common.bit_count()} common neighbors")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
