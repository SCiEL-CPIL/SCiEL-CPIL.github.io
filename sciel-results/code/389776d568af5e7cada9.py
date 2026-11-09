"""Exact XOR-circuit synthesis with Recursive Clustering Divide and Conquer."""

def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict

    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Wire numbering: 0..n-1 are inputs, n is constant zero
    # Gates produce wires n+1, n+2, ...
    
    gates = []
    next_wire = n + 1
    
    def new_gate(a, b):
        nonlocal next_wire
        gates.append([int(a), int(b)])
        w = next_wire
        next_wire += 1
        return w

    def solve_submatrix(sub_matrix, input_indices):
        """
        Solves the sub-problem defined by sub_matrix (m x k) and input_indices (list of global input indices).
        Returns:
            outputs: list of global wire indices for the m outputs.
        """
        k = len(input_indices)
        
        # Base case: small k
        if k <= 1:
            outputs = []
            for row in sub_matrix:
                if k == 0:
                    outputs.append(n) # Zero
                elif k == 1:
                    if row[0] == 1:
                        outputs.append(input_indices[0])
                    else:
                        outputs.append(n)
            return outputs

        # If k is small enough, use a simple greedy CSE locally
        # This avoids deep recursion overhead and handles local sharing well
        if k <= 6:
            return solve_local_greedy(sub_matrix, input_indices)

        # Recursive Step
        # 1. Compute co-occurrence matrix for local inputs
        # sub_matrix is m x k
        # Co-occurrence C[i][j] = sum(row_i & row_j) ? No, sum over rows of (row[i] & row[j])
        C = np.zeros((k, k), dtype=int)
        # Efficient calculation: C = sub_matrix.T @ sub_matrix
        C = sub_matrix.T @ sub_matrix
        
        # 2. Partition indices 0..k-1 into two groups G1, G2
        # Heuristic: Greedy partitioning to minimize cross co-occurrence
        # Or simple split if k is even?
        # Let's try a simple greedy:
        # Start with G1=[0], G2=[]
        # For each remaining index, add to G1 or G2 that minimizes increase in cross-cost.
        
        G1 = [0]
        G2 = []
        remaining = list(range(1, k))
        
        # Precompute row sums of C for faster updates?
        # cost1 = sum(C[i][j] for i in G1, j in G2)
        # We want to minimize this.
        
        # Simple heuristic: Sort by total co-occurrence (degree) and split?
        # Or just split in half?
        # Let's try a simple greedy:
        current_cost = 0
        # Sum of C[i] for i in G1
        sum_G1 = C[0].copy()
        sum_G2 = np.zeros(k, dtype=int)
        
        for idx in remaining:
            # Cost if added to G1: sum(C[idx][j] for j in G2) -> dot(C[idx], sum_G2)
            cost_add_G1 = np.dot(C[idx], sum_G2)
            # Cost if added to G2: sum(C[idx][j] for j in G1) -> dot(C[idx], sum_G1)
            cost_add_G2 = np.dot(C[idx], sum_G1)
            
            if cost_add_G1 < cost_add_G2:
                G1.append(idx)
                sum_G1 += C[idx]
                current_cost += cost_add_G1
            else:
                G2.append(idx)
                sum_G2 += C[idx]
                current_cost += cost_add_G2
        
        # Create sub-matrices
        sub1 = sub_matrix[:, G1]
        sub2 = sub_matrix[:, G2]
        
        # Map global input indices
        in1 = [input_indices[i] for i in G1]
        in2 = [input_indices[i] for i in G2]
        
        # Recurse
        out1 = solve_submatrix(sub1, in1)
        out2 = solve_submatrix(sub2, in2)
        
        # Combine
        final_outputs = []
        for w1, w2 in zip(out1, out2):
            if w1 == n and w2 == n:
                final_outputs.append(n)
            elif w1 == n:
                final_outputs.append(w2)
            elif w2 == n:
                final_outputs.append(w1)
            elif w1 == w2:
                # w1 XOR w1 = 0
                final_outputs.append(n)
            else:
                final_outputs.append(new_gate(w1, w2))
                
        return final_outputs

    def solve_local_greedy(sub_matrix, input_indices):
        """
        Solves a small sub-matrix using greedy CSE.
        Returns list of global wire indices.
        """
        k = len(input_indices)
        m_sub = sub_matrix.shape[0]
        
        # Map local indices 0..k-1 to global input indices
        # We will work with local supports, then map to global wires at the end?
        # No, we need to create global gates.
        
        # Local supports: list of sets of local indices
        supports = []
        for row in sub_matrix:
            indices = set(np.where(row > 0)[0].tolist())
            supports.append(indices)
            
        # Local wire mapping:
        # 0..k-1 map to input_indices[0..k-1]
        # k maps to n (zero)
        # k+1.. are new gates
        
        local_gates = [] # List of [local_wire1, local_wire2]
        
        def local_new_gate(a, b):
            local_gates.append([a, b])
            return k + len(local_gates) # Local wire index
        
        # Greedy CSE on local indices
        while True:
            freq = defaultdict(int)
            for sup in supports:
                if len(sup) < 2:
                    continue
                items = list(sup)
                for i in range(len(items)):
                    for j in range(i+1, len(items)):
                        u, v = items[i], items[j]
                        if u > v:
                            u, v = v, u
                        freq[(u, v)] += 1
                        
            max_freq = 0
            best_pair = None
            for pair, count in freq.items():
                if count > max_freq:
                    max_freq = count
                    best_pair = pair
                    
            if max_freq < 2:
                break
                
            u, v = best_pair
            w = local_new_gate(u, v)
            
            for i, sup in enumerate(supports):
                if u in sup and v in sup:
                    sup.remove(u)
                    sup.remove(v)
                    sup.add(w)
                    
        # Convert local gates to global gates
        # We need to map local wire indices to global wire indices.
        # Local 0..k-1 -> input_indices[0..k-1]
        # Local k -> n
        # Local k+1+i -> new global wire
        
        local_to_global = {}
        for i in range(k):
            local_to_global[i] = input_indices[i]
        local_to_global[k] = n
        
        global_outputs = []
        for i in range(len(local_gates)):
            lu, lv = local_gates[i]
            gu = local_to_global.get(lu)
            gv = local_to_global.get(lv)
            if gu is None or gv is None:
                # This shouldn't happen if we process in order, but local gates refer to previous local gates
                # We must resolve them as we go?
                # Actually, local_gates are generated in order.
                # The wire produced by local gate i is local index k+1+i.
                # So when we process gate i, its inputs u, v must be either < k+1+i.
                # So local_to_global should already have entries for u and v.
                pass
            
            gw = new_gate(gu, gv)
            local_to_global[k + 1 + i] = gw
            
        # Map final supports to global wires
        final_outputs = []
        for sup in supports:
            if len(sup) == 0:
                final_outputs.append(n)
            elif len(sup) == 1:
                w = list(sup)[0]
                final_outputs.append(local_to_global[w])
            else:
                wires = [local_to_global[w] for w in sup]
                result = wires[0]
                for w in wires[1:]:
                    result = new_gate(result, w)
                final_outputs.append(result)
                
        return final_outputs

    # Main call
    # input_indices is 0..n-1
    initial_inputs = list(range(n))
    outputs = solve_submatrix(matrix, initial_inputs)
    
    return {'gates': gates, 'outputs': outputs}
