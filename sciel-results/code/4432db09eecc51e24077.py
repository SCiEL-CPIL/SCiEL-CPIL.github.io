import random
import time
import itertools

def construct():
    n = 34
    
    # Precomputation
    triples = list(itertools.combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    # Identify K4s and K5s
    k4_edges = []
    for quad in itertools.combinations(range(n), 4):
        edges = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
        k4_edges.append(edges)
        
    k5_edges = []
    for quint in itertools.combinations(range(n), 5):
        edges = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
        k5_edges.append(edges)
        
    num_k4 = len(k4_edges)
    num_k5 = len(k5_edges)
    
    # Map edge to constraints
    edge_to_k4s = [[] for _ in range(num_triples)]
    edge_to_k5s = [[] for _ in range(num_triples)]
    
    for i, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(i)
            
    for i, edges in enumerate(k5_edges):
        for e in edges:
            edge_to_k5s[e].append(i)
            
    def solve_local_search(time_limit=110):
        start_time = time.time()
        
        # Try multiple restarts
        for restart in range(20):
            # Initialize with random colors
            random.seed(1000 + restart)
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Initialize counts
            red_mono = [0] * num_k4
            blue_mono = [0] * num_k5
            
            for i in range(num_k4):
                red_mono[i] = sum(1 for e in k4_edges[i] if colors[e] == 0)
                
            for i in range(num_k5):
                blue_mono[i] = sum(1 for e in k5_edges[i] if colors[e] == 1)
                
            current_cost = sum(1 for c in red_mono if c == 4) + sum(1 for c in blue_mono if c == 10)
            
            if current_cost == 0:
                return colors
                
            no_improve = 0
            max_no_improve = 5000
            
            while time.time() - start_time < time_limit:
                if current_cost == 0:
                    return colors
                
                # Pick a random edge to flip
                e = random.randint(0, num_triples - 1)
                old_color = colors[e]
                new_color = 1 - old_color
                
                delta_cost = 0
                
                # Update Red K4s
                for k4_idx in edge_to_k4s[e]:
                    old_count = red_mono[k4_idx]
                    if old_color == 0:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                        
                    old_viol = 1 if old_count == 4 else 0
                    new_viol = 1 if new_count == 4 else 0
                    delta_cost += new_viol - old_viol
                    
                    # Tentatively update count
                    red_mono[k4_idx] = new_count
                    
                # Update Blue K5s
                for k5_idx in edge_to_k5s[e]:
                    old_count = blue_mono[k5_idx]
                    if old_color == 1:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                        
                    old_viol = 1 if old_count == 10 else 0
                    new_viol = 1 if new_count == 10 else 0
                    delta_cost += new_viol - old_viol
                    
                    # Tentatively update count
                    blue_mono[k5_idx] = new_count
                    
                colors[e] = new_color
                new_cost = current_cost + delta_cost
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                elif new_cost == current_cost:
                    if random.random() < 0.5:
                        no_improve += 1
                    else:
                        # Rollback
                        colors[e] = old_color
                        for k4_idx in edge_to_k4s[e]:
                            if old_color == 0:
                                red_mono[k4_idx] += 1
                            else:
                                red_mono[k4_idx] -= 1
                        for k5_idx in edge_to_k5s[e]:
                            if old_color == 1:
                                blue_mono[k5_idx] += 1
                            else:
                                blue_mono[k5_idx] -= 1
                        no_improve += 1
                else:
                    # Rollback
                    colors[e] = old_color
                    for k4_idx in edge_to_k4s[e]:
                        if old_color == 0:
                            red_mono[k4_idx] += 1
                        else:
                            red_mono[k4_idx] -= 1
                    for k5_idx in edge_to_k5s[e]:
                        if old_color == 1:
                            blue_mono[k5_idx] += 1
                        else:
                            blue_mono[k5_idx] -= 1
                    no_improve += 1
                    
                if no_improve > max_no_improve:
                    break
            
        return None

    colors = solve_local_search()
    
    if colors is not None:
        return {'n': 34, 'colors': colors}
    else:
        # Fallback: Return a valid small n
        # We need a guaranteed valid coloring for small n.
        # n=1,2,3,4,5,6,7,8,9...
        # For n=8, C(8,4)=70 K4s, C(8,5)=56 K5s.
        # Random search for n=8 is very fast.
        # Let's just try to find a valid n=8 coloring if n=34 fails.
        
        def find_valid_small_n(n_target, time_limit=5):
            triples_s = list(itertools.combinations(range(n_target), 3))
            triple_to_idx_s = {t: i for i, t in enumerate(triples_s)}
            num_triples_s = len(triples_s)
            
            k4_edges_s = []
            for quad in itertools.combinations(range(n_target), 4):
                edges = [triple_to_idx_s[t] for t in itertools.combinations(quad, 3)]
                k4_edges_s.append(edges)
                
            k5_edges_s = []
            for quint in itertools.combinations(range(n_target), 5):
                edges = [triple_to_idx_s[t] for t in itertools.combinations(quint, 3)]
                k5_edges_s.append(edges)
                
            num_k4_s = len(k4_edges_s)
            num_k5_s = len(k5_edges_s)
            
            edge_to_k4s_s = [[] for _ in range(num_triples_s)]
            edge_to_k5s_s = [[] for _ in range(num_triples_s)]
            
            for i, edges in enumerate(k4_edges_s):
                for e in edges:
                    edge_to_k4s_s[e].append(i)
                    
            for i, edges in enumerate(k5_edges_s):
                for e in edges:
                    edge_to_k5s_s[e].append(i)
                    
            start_time = time.time()
            while time.time() - start_time < time_limit:
                random.seed()
                colors_s = [random.randint(0, 1) for _ in range(num_triples_s)]
                red_mono_s = [0] * num_k4_s
                blue_mono_s = [0] * num_k5_s
                
                for i in range(num_k4_s):
                    red_mono_s[i] = sum(1 for e in k4_edges_s[i] if colors_s[e] == 0)
                for i in range(num_k5_s):
                    blue_mono_s[i] = sum(1 for e in k5_edges_s[i] if colors_s[e] == 1)
                
                cost = sum(1 for c in red_mono_s if c == 4) + sum(1 for c in blue_mono_s if c == 10)
                
                if cost == 0:
                    return colors_s
                    
                # Simple hill climb
                no_imp = 0
                while time.time() - start_time < time_limit and cost > 0:
                    e = random.randint(0, num_triples_s - 1)
                    old_c = colors_s[e]
                    new_c = 1 - old_c
                    delta = 0
                    for k4_i in edge_to_k4s_s[e]:
                        oc = red_mono_s[k4_i]
                        nc = oc - 1 if old_c == 0 else oc + 1
                        delta += (1 if nc == 4 else 0) - (1 if oc == 4 else 0)
                        red_mono_s[k4_i] = nc
                    for k5_i in edge_to_k5s_s[e]:
                        oc = blue_mono_s[k5_i]
                        nc = oc - 1 if old_c == 1 else oc + 1
                        delta += (1 if nc == 10 else 0) - (1 if oc == 10 else 0)
                        blue_mono_s[k5_i] = nc
                        
                    colors_s[e] = new_c
                    new_cost = cost + delta
                    if new_cost < cost:
                        cost = new_cost
                        no_imp = 0
                    else:
                        # Rollback
                        colors_s[e] = old_c
                        for k4_i in edge_to_k4s_s[e]:
                            if old_c == 0: red_mono_s[k4_i] += 1
                            else: red_mono_s[k4_i] -= 1
                        for k5_i in edge_to_k5s_s[e]:
                            if old_c == 1: blue_mono_s[k5_i] += 1
                            else: blue_mono_s[k5_i] -= 1
                        no_imp += 1
                        if no_imp > 1000: break
                        
            return None

        colors_fallback = find_valid_small_n(9)
        if colors_fallback:
            return {'n': 9, 'colors': colors_fallback}
        else:
            # Last resort: n=1
            return {'n': 1, 'colors': []}
