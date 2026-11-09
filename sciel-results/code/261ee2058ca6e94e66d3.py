import time
import random
import math
from itertools import combinations

def _has_k4(adj, n):
    """Check if graph has a K4. adj is a list of sets."""
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                continue
            for k in range(j+1, n):
                if k not in adj[i] or k not in adj[j]:
                    continue
                for l in range(k+1, n):
                    if l in adj[i] and l in adj[j] and l in adj[k]:
                        return True
    return False

def _has_independent_6(adj, n):
    """Check if graph has an independent set of size 6."""
    # Independent set in G = clique in complement
    # Complement adjacency: i~j iff j not in adj[i]
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                continue
            for k in range(j+1, n):
                if k in adj[i] or k in adj[j]:
                    continue
                for l in range(k+1, n):
                    if l in adj[i] or l in adj[j] or l in adj[k]:
                        continue
                    for m in range(l+1, n):
                        if m in adj[i] or m in adj[j] or m in adj[k] or m in adj[l]:
                            continue
                        for p in range(m+1, n):
                            if (p in adj[i] or p in adj[j] or p in adj[k] or 
                                p in adj[l] or p in adj[m]):
                                continue
                            return True
    return False

def _is_valid(adj, n):
    return not _has_k4(adj, n) and not _has_independent_6(adj, n)

def _count_k4s(adj, n):
    """Count K4s for penalty."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                continue
            for k in range(j+1, n):
                if k not in adj[i] or k not in adj[j]:
                    continue
                for l in range(k+1, n):
                    if l in adj[i] and l in adj[j] and l in adj[k]:
                        count += 1
    return count

def _count_indep6(adj, n):
    """Count independent 6-sets for penalty."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                continue
            for k in range(j+1, n):
                if k in adj[i] or k in adj[j]:
                    continue
                for l in range(k+1, n):
                    if l in adj[i] or l in adj[j] or l in adj[k]:
                        continue
                    for m in range(l+1, n):
                        if m in adj[i] or m in adj[j] or m in adj[k] or m in adj[l]:
                            continue
                        for p in range(m+1, n):
                            if (p in adj[i] or p in adj[j] or p in adj[k] or 
                                p in adj[l] or p in adj[m]):
                                continue
                            count += 1
    return count

def _paley_graph(n):
    """Construct a Paley-like graph. For prime p ≡ 1 mod 4, use quadratic residues."""
    # Find smallest prime >= n that is ≡ 1 mod 4
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0 or x % 3 == 0: return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i+2) == 0:
                return False
            i += 6
        return True
    
    p = n
    while not is_prime(p) or p % 4 != 1:
        p += 1
    
    if p > 40:
        return None
    
    squares = set()
    for x in range(1, p):
        squares.add(x*x % p)
    
    adj = [set() for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j-i) % p in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    edges = []
    for i in range(p):
        for j in sorted(adj[i]):
            if j > i:
                edges.append([i, j])
    
    return {'n': p, 'edges': edges}

def _random_graph(n, density):
    """Generate a random graph with given edge density."""
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < density:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    return adj, edges

def _simulated_annealing(n, density, max_iter, start_temp=10.0, cool_rate=0.999):
    """Simulated annealing to find a valid K4-free graph with no independent 6-set."""
    adj, edges = _random_graph(n, density)
    
    def penalty(adj):
        k4_count = _count_k4s(adj, n)
        indep6_count = _count_indep6(adj, n)
        return k4_count * 10 + indep6_count * 10
    
    current_pen = penalty(adj)
    best_adj = [s.copy() for s in adj]
    best_pen = current_pen
    
    temp = start_temp
    for it in range(max_iter):
        if temp < 0.01:
            break
        
        # Pick a random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        
        # Try flipping edge (i,j)
        if j in adj[i]:
            adj[i].remove(j)
            adj[j].remove(i)
        else:
            adj[i].add(j)
            adj[j].add(i)
        
        new_pen = penalty(adj)
        delta = new_pen - current_pen
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_pen = new_pen
            if new_pen < best_pen:
                best_pen = new_pen
                best_adj = [s.copy() for s in adj]
                if best_pen == 0:
                    break
        else:
            # Revert
            if j in adj[i]:
                adj[i].remove(j)
                adj[j].remove(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
        
        temp *= cool_rate
    
    if best_pen == 0:
        edges = []
        for i in range(n):
            for j in sorted(best_adj[i]):
                if j > i:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

def _local_search(adj, n, max_iter=5000):
    """Local search: try to eliminate K4s and independent 6-sets by edge flips."""
    def penalty(adj):
        k4_count = _count_k4s(adj, n)
        indep6_count = _count_indep6(adj, n)
        return k4_count * 10 + indep6_count * 10
    
    current_pen = penalty(adj)
    for it in range(max_iter):
        if current_pen == 0:
            break
        
        best_move = None
        best_pen = current_pen
        
        # Try some random edge flips
        for _ in range(20):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            old_state = j in adj[i]
            if old_state:
                adj[i].remove(j)
                adj[j].remove(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
            
            new_pen = penalty(adj)
            
            if new_pen < best_pen:
                best_pen = new_pen
                best_move = (i, j, not old_state)
            else:
                # Revert
                if old_state:
                    adj[i].add(j)
                    adj[j].add(i)
                else:
                    adj[i].remove(j)
                    adj[j].remove(i)
        
        if best_move is None:
            break
        
        i, j, add = best_move
        if add:
            adj[i].add(j)
            adj[j].add(i)
        else:
            adj[i].remove(j)
            adj[j].remove(i)
        current_pen = best_pen
    
    if current_pen == 0:
        edges = []
        for i in range(n):
            for j in sorted(adj[i]):
                if j > i:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try Paley graphs first (known good constructions)
    for p in [17, 29, 37]:
        if p > 40:
            break
        result = _paley_graph(p)
        if result is not None:
            adj = [set() for _ in range(result['n'])]
            for u, v in result['edges']:
                adj[u].add(v)
                adj[v].add(u)
            if _is_valid(adj, result['n']):
                return result
        if time.time() > deadline:
            break
    
    # Try to build larger graphs via simulated annealing
    best_result = None
    best_n = 0
    
    # Try n from high to low
    for n in range(35, 16, -1):
        if time.time() > deadline - 5:
            break
        
        # Multiple restarts with different densities
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            density = 0.3 + 0.1 * restart
            result = _simulated_annealing(n, density, max_iter=500, start_temp=5.0, cool_rate=0.9995)
            if result is not None:
                adj = [set() for _ in range(n)]
                for u, v in result['edges']:
                    adj[u].add(v)
                    adj[v].add(u)
                if _is_valid(adj, n):
                    if n > best_n:
                        best_n = n
                        best_result = result
                    if n >= 35:
                        return result
    
    # If we found something, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
