"""Construct 4-regular graph with girth 10 using Lazebnik-Ustimenko-Woldar construction."""
import numpy as np
from collections import deque

def construct(k, g):
    q = k
    d = max(3, g - 5)
    if d % 2 == 0:
        d += 1
    
    if q == 4:
        mult_table = np.array([
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ], dtype=np.int32)
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: mult_table[x, y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    total_coords = d + 1
    q_pow_total = q ** total_coords
    q_powers = [q ** i for i in range(total_coords)]
    
    def encode(side, coords_list):
        idx = side * q_pow_total
        for i in range(1, total_coords):
            idx += coords_list[i] * q_powers[i]
        return idx
    
    def decode(idx):
        side = idx // q_pow_total
        rem = idx % q_pow_total
        coords = [0] * total_coords
        for i in range(total_coords - 1, 0, -1):
            coords[i] = rem % q
            rem //= q
        return side, coords
    
    start_coords = [0] * total_coords
    start_idx = encode(0, start_coords)
    
    visited = {start_idx}
    edge_set = set()
    queue = deque([start_idx])
    
    while queue:
        v = queue.popleft()
        side, coords = decode(v)
        
        for first in range(q):
            if side == 0:
                other_coords = [0] * total_coords
                other_coords[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    other_coords[c] = add_op(coords[c], mul_op(other_coords[a], coords[b]))
                other_side = 1
            else:
                other_coords = [0] * total_coords
                other_coords[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    other_coords[c] = sub_op(coords[c], mul_op(coords[a], other_coords[b]))
                other_side = 0
            
            other_idx = encode(other_side, other_coords)
            
            edge_key = (min(v, other_idx), max(v, other_idx))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
            
            if other_idx not in visited:
                visited.add(other_idx)
                queue.append(other_idx)
    
    n = len(visited)
    sorted_verts = sorted(visited)
    index_map = {v: i for i, v in enumerate(sorted_verts)}
    
    edges = []
    for e in sorted(edge_set):
        edges.append((index_map[e[0]], index_map[e[1]]))
    
    return n, edges
