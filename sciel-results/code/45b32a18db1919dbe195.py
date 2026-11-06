import time
import numpy as np
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    A = np.array(matrix, dtype=np.uint8) % 2
    
    # Compute RREF with row tracking
    M = A.copy()
    T = np.eye(m, dtype=np.uint8)
    
    rank = 0
    pivot_cols = []
    
    for col in range(n):
        if time.time() > deadline:
            break
            
        # Heuristic: Choose the pivot row with the minimum popcount to keep basis vectors sparse
        pivot_candidates = [row for row in range(rank, m) if M[row, col] == 1]
        if not pivot_candidates:
            continue
        
        best_pivot = min(pivot_candidates, key=lambda r: np.sum(M[r]))
        
        M[[rank, best_pivot]] = M[[best_pivot, rank]]
        T[[rank, best_pivot]] = T[[best_pivot, rank]]
        
        for row in range(m):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                T[row] ^= T[rank]
        
        pivot_cols.append(col)
        rank += 1
        if rank == m:
            break
    
    # Compute T^{-1}
    aug = np.hstack([T, np.eye(m, dtype=np.uint8)])
    for col in range(m):
        if time.time() > deadline:
            break
        pivot = -1
        for row in range(col, m):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(m):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    T_inv = aug[:, m:]
    
    # Build XOR trees efficiently using a helper that reuses intermediate results
    gates = []
    wire_cache = {}  # frozenset of inputs -> wire number
    
    def xor_wires(wire_list):
        """XOR a list of wires, reusing cached intermediate results."""
        if not wire_list:
            return n
        if len(wire_list) == 1:
            return wire_list[0]
        
        key = frozenset(wire_list)
        if key in wire_cache:
            return wire_cache[key]
        
        # Use divide-and-conquer to build balanced XOR tree
        mid = len(wire_list) // 2
        left = xor_wires(wire_list[:mid])
        right = xor_wires(wire_list[mid:])
        
        g = len(gates)
        gates.append([left, right])
        result = n + 1 + g
        wire_cache[key] = result
        return result
    
    # Step 1: For each pivot row, compute the wire
    pivot_wires = []
    for i in range(rank):
        if time.time() > deadline:
            # Fallback: simple linear XOR
            ones = [j for j in range(n) if M[i, j] == 1]
            if not ones:
                pivot_wires.append(n)
            elif len(ones) == 1:
                pivot_wires.append(ones[0])
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    g = len(gates)
                    gates.append([current, ones[j]])
                    current = n + 1 + g
                pivot_wires.append(current)
        else:
            ones = [j for j in range(n) if M[i, j] == 1]
            pivot_wires.append(xor_wires(ones))
    
    # Step 2: Express each output as XOR of pivot wires
    outputs = []
    for j in range(m):
        if time.time() > deadline:
            # Fallback: simple linear XOR
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            if not contributing:
                outputs.append(n)
            elif len(contributing) == 1:
                outputs.append(pivot_wires[contributing[0]])
            else:
                current = pivot_wires[contributing[0]]
                for i in range(1, len(contributing)):
                    g = len(gates)
                    gates.append([current, pivot_wires[contributing[i]]])
                    current = n + 1 + g
                outputs.append(current)
        else:
            contributing = [i for i in range(rank) if T_inv[j, i] == 1]
            outputs.append(xor_wires([pivot_wires[i] for i in contributing]))
    
    return {'gates': gates, 'outputs': outputs}
