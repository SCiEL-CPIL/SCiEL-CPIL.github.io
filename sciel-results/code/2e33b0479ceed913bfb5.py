import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to integers
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    # Unique targets
    unique_targets = list(set(rows))
    
    # If n is small, we can use BFS on the hypercube
    # n is max 16, so 2^16 = 65536 states
    if n <= 20: # Safety check, though n is small
        size = 1 << n
        dist = [-1] * size
        parent = [-1] * size
        parent_bit = [-1] * size
        
        # Initialize with inputs and zero
        # Inputs are e_0, ..., e_{n-1}
        # Zero is 0
        queue = []
        
        # Add zero
        dist[0] = 0
        queue.append(0)
        
        # Add inputs
        for i in range(n):
            val = 1 << i
            dist[val] = 0
            queue.append(val)
            
        # BFS
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            
            for j in range(n):
                v = u ^ (1 << j)
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    parent_bit[v] = j
                    queue.append(v)
                    
        # Now we have dist and parent for all vectors.
        # We want to construct wires for all unique targets.
        # Sort targets by dist to maximize sharing of prefixes?
        # Actually, we should process targets such that we build the necessary intermediates.
        
        val_to_wires = {}
        gates = []
        next_wire = n + 1
        
        # Initialize wires for inputs and zero
        for i in range(n):
            val_to_wires[1 << i] = i
        val_to_wires[0] = n
        
        # Sort unique targets by distance
        # If dist is -1 (should not happen), skip or handle
        sorted_targets = sorted([t for t in unique_targets if t != 0], key=lambda x: dist[x])
        
        for t in sorted_targets:
            if t in val_to_wires:
                continue
            
            # Trace path from t back to a source (dist 0)
            path = []
            curr = t
            while dist[curr] > 0:
                path.append(curr)
                curr = parent[curr]
            # curr is now a source (input or zero)
            
            # Process path in reverse (source to target)
            for v in reversed(path):
                if v in val_to_wires:
                    continue
                
                p = parent[v]
                bit = parent_bit[v]
                
                # Gate: wire[p] XOR input[bit] -> wire[v]
                # Check if p is in val_to_wires
                # It should be, because we process in topological order of the path
                # But wait, the path is a tree. If p was not on the path of a previously
                # processed target, is it available?
                # Yes, because we trace back to a source.
                # However, if p is an intermediate that was NOT created yet, we have a problem.
                # But we are processing the path from source to target.
                # So when we reach v, p is closer to the source.
                # If p was not created by a previous target, we must create it now?
                # No, the loop `for v in reversed(path)` iterates from source-side to target-side.
                # So when we process v, we assume p is available.
                # Is p available?
                # If p is a source, yes.
                # If p is an intermediate, it was earlier in the `reversed(path)` list.
                # So if p was not in `val_to_wires` initially, we would have created it in the previous iteration of this loop.
                
                gates.append([val_to_wires[p], bit])
                val_to_wires[v] = next_wire
                next_wire += 1
                
        # Map original rows to wires
        outputs = [val_to_wires[row] for row in rows]
        
        return {'gates': gates, 'outputs': outputs}
    else:
        # Fallback for large n (though n is small in this task)
        # Simple greedy
        pass
