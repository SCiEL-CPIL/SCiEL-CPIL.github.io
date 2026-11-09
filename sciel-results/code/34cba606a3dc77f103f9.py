"""
Program to construct a C4-free graph with no independent set of size 11, maximizing n.
Uses a polarity graph base (PG(2,5)) and randomized greedy extension with local search.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)

def _popcount(x):
    return bin(x).count('1')

def _has_c4(adj, adj_bits, n):
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            if common & (common - 1):
                return True
    return False

def _can_add_edge(adj, adj_bits, u, v, n):
    for x in adj[v]:
        if x == u:
            continue
        if adj_bits[u] & adj_bits[x]:
            return False
    for y in adj[u]:
        if y == v:
            continue
        if adj_bits[v] & adj_bits[y]:
            return False
    return True

def _add_edge(adj, adj_bits, u, v):
    adj[u].add(v)
    adj[v].add(u)
    adj_bits[u] |= (1 << v)
    adj_bits[v] |= (1 << u)

def _remove_edge(adj, adj_bits, u, v):
    adj[u].discard(v)
    adj[v].discard(u)
    adj_bits[u] &= ~(1 << v)
    adj_bits[v] &= ~(1 << u)

def _has_independent_set_k(adj_bits, n, k):
    found = False
    
    def search(rem, size):
        nonlocal found
        if found:
            return
        if size >= k:
            found = True
            return
        if rem == 0:
            return
        if size + _popcount(rem) < k:
            return
            
        best_v = -1
        best_deg = -1
        temp_rem = rem
        while temp_rem:
            lsb = temp_rem & -temp_rem
            v = lsb.bit_length() - 1
            deg = _popcount(adj_bits[v] & rem)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_rem ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        new_rem = rem & ~(adj_bits[best_v] | v_bit)
        search(new_rem, size + 1)
        if found:
            return
            
        new_rem2 = rem & ~v_bit
        search(new_rem2, size)

    search((1 << n) - 1, 0)
    return found

def _polarity_graph_pg25():
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else:
                    inv_z = pow(z, 3, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                points.append((nx, ny, nz))
    
    unique_points = list(set(points))
    unique_points.sort()
    n = len(unique_points)
    
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = unique_points[i]
            p2 = unique_points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                _add_edge(adj, adj_bits, i, j)
                
    return n, adj, adj_bits

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n0, adj, adj_bits = _polarity_graph_pg25()
    n = n0
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    if _has_c4(adj, adj_bits, n):
        pass
    if _has_independent_set_k(adj_bits, n, 11):
        pass
        
    rng = random.Random(42)
    max_n = 43
    
    def get_greedy_max_independent_set(adj_bits, n):
        rem = (1 << n) - 1
        S = set()
        while rem:
            best_v = -1
            min_deg = n
            temp_rem = rem
            while temp_rem:
                lsb = temp_rem & -temp_rem
                v = lsb.bit_length() - 1
                deg = _popcount(adj_bits[v] & rem)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp_rem ^= lsb
            if best_v == -1:
                break
            S.add(best_v)
            rem &= ~(adj_bits[best_v] | (1 << best_v))
        return S

    for target_n in range(n0 + 1, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        success = False
        target_is = get_greedy_max_independent_set(adj_bits, n)
        
        for r in range(200):
            if time.time() > deadline - 5:
                break
                
            adj_copy = [set(s) for s in adj]
            adj_bits_copy = list(adj_bits)
            
            v_new = n
            
            in_target = [v for v in range(n) if v in target_is]
            not_in_target = [v for v in range(n) if v not in target_is]
            
            rng.shuffle(in_target)
            rng.shuffle(not_in_target)
            
            S = set()
            
            for u in in_target:
                can_add = True
                for w in S:
                    if adj_bits_copy[u] & adj_bits_copy[w]:
                        can_add = False
                        break
                if can_add:
                    S.add(u)
                    
            for u in not_in_target:
                can_add = True
                for w in S:
                    if adj_bits_copy[u] & adj_bits_copy[w]:
                        can_add = False
                        break
                if can_add:
                    S.add(u)
            
            if not S:
                vertices = list(range(n))
                rng.shuffle(vertices)
                for u in vertices:
                    can_add = True
                    for w in S:
                        if adj_bits_copy[u] & adj_bits_copy[w]:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)

            adj_copy.append(set(S))
            adj_bits_copy.append(0)
            new_bits = 0
            for u in S:
                adj_copy[u].add(v_new)
                adj_bits_copy[u] |= (1 << v_new)
                new_bits |= (1 << u)
            adj_bits_copy[v_new] = new_bits
            
            if not _has_independent_set_k(adj_bits_copy, n + 1, 11):
                adj = adj_copy
                adj_bits = adj_bits_copy
                n += 1
                success = True
                break
        
        if not success:
            break
        
        best_n = n
        best_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    best_edges.append([u, v])

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
