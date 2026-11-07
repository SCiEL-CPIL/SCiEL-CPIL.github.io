<<<<<<< SEARCH
    def build_matrix(signs):
        """Build signed adjacency matrix from signs."""
        A = np.zeros((n, n), dtype=np.float64)
        A[edge_u, edge_v] = signs.astype(np.float64)
        A[edge_v, edge_u] = signs.astype(np.float64)
        return A
    
    def compute_radius_and_eigvector(signs):
        """Compute spectral radius and corresponding eigenvector."""
        A = build_matrix(signs)
        eigvals, eigvecs = np.linalg.eigh(A)
        # Find eigenvalue with largest absolute value
        idx = np.argmax(np.abs(eigvals))
        radius = abs(eigvals[idx])
        eigvec = eigvecs[:, idx]
        return radius, eigvec
    
    def is_connected_lift(signs):
        """
        Check if the 2-lift is connected using BFS.
        The lift has 2*n vertices: (v, 0) and (v, 1).
        Edge (u, v) with sign s connects:
          - (u, 0) <-> (v, 0) and (u, 1) <-> (v, 1) if s = +1
          - (u, 0) <-> (v, 1) and (u, 1) <-> (v, 0) if s = -1
        """
        # Build lift adjacency
        lift_adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Same side connections
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # Opposite side connections
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # BFS from vertex 0
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == 2 * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 58.0  # 58 seconds safety margin (90s limit)
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    def local_search(initial_signs, max_iter=10000):
        """
        Local search: flip individual signs to reduce spectral radius.
        Uses gradient information to prioritize flips.
        """
        signs = initial_signs.copy()
        current_radius, current_eigvec = compute_radius_and_eigvector(signs)
        
        if not is_connected_lift(signs):
            return None, float('inf')
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            best_flip = None
            best_new_radius = current_radius
            
            # Compute gradient of spectral radius w.r.t. each sign
            # d(radius)/d(sign_i) = 2 * A[i,j] * eigvec[i] * eigvec[j] for the dominant eigenvector
            # Since flipping sign_i changes A[u,v] and A[v,u] from s to -s,
            # the change in radius is approximately 2*s*|eigvec[u]*eigvec[v]|
            
            # Try all flips, but use gradient to order them
            flip_order = list(range(num_edges))
            
            # Compute estimated improvement for each flip
            A = build_matrix(signs)
            improvements = np.zeros(num_edges)
            for idx in range(num_edges):
                u, v = edge_u[idx], edge_v[idx]
                s = signs[idx]
                # Change in A: A[u,v] goes from s to -s, change = -2*s
                # Approximate change in lambda_max = eigvec^T * delta_A * eigvec
                delta = -2 * s * (current_eigvec[u] * current_eigvec[v] + current_eigvec[v] * current_eigvec[u])
                improvements[idx] = -delta  # We want to minimize radius, so negative delta is good
            
            # Sort by estimated improvement (most negative first = best)
            flip_order.sort(key=lambda i: improvements[i])
            
            # Try flips in order of estimated improvement
            for idx in flip_order:
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_radius, new_eigvec = compute_radius_and_eigvector(signs)
                signs[idx] = old_sign
                
                if new_radius < best_new_radius - 1e-12:
                    best_new_radius = new_radius
                    best_flip = idx
            
            if best_flip is None:
                break
            
            signs[best_flip] = -signs[best_flip]
            current_radius, current_eigvec = compute_radius_and_eigvector(signs)
            
            # Check connectivity periodically
            if iteration % 10 == 0 and not is_connected_lift(signs):
                # Revert the flip that broke connectivity
                signs[best_flip] = -signs[best_flip]
                current_radius, current_eigvec = compute_radius_and_eigvector(signs)
                break
        
        if is_connected_lift(signs):
            return signs, current_radius
        else:
            return None, float('inf')
=======
    def build_matrix(signs):
        """Build signed adjacency matrix from signs."""
        A = np.zeros((n, n), dtype=np.float64)
        A[edge_u, edge_v] = signs.astype(np.float64)
        A[edge_v, edge_u] = signs.astype(np.float64)
        return A
    
    def compute_radius_and_eigvector(signs):
        """Compute spectral radius and corresponding eigenvector."""
        A = build_matrix(signs)
        eigvals, eigvecs = np.linalg.eigh(A)
        # Find eigenvalue with largest absolute value
        idx = np.argmax(np.abs(eigvals))
        radius = abs(eigvals[idx])
        eigvec = eigvecs[:, idx]
        return radius, eigvec
    
    def is_connected_lift(signs):
        """
        Check if the 2-lift is connected using BFS.
        """
        lift_adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        visited = [False] * (2 * n)
        queue = deque([0])
        visited[0] = True
        count = 1
        while queue:
            node = queue.popleft()
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == 2 * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 58.0
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    def local_search(initial_signs, max_iter=10000):
        """
        Local search: flip individual signs to reduce spectral radius.
        """
        signs = initial_signs.copy()
        current_radius, current_eigvec = compute_radius_and_eigvector(signs)
        
        if not is_connected_lift(signs):
            return None, float('inf')
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            best_flip = None
            best_new_radius = current_radius
            
            flip_order = list(range(num_edges))
            
            # Compute estimated improvement for each flip
            improvements = np.zeros(num_edges)
            for idx in range(num_edges):
                u, v = edge_u[idx], edge_v[idx]
                s = signs[idx]
                delta = -2 * s * (current_eigvec[u] * current_eigvec[v] + current_eigvec[v] * current_eigvec[u])
                improvements[idx] = -delta
            
            flip_order.sort(key=lambda i: improvements[i])
            
            for idx in flip_order:
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_radius, new_eigvec = compute_radius_and_eigvector(signs)
                signs[idx] = old_sign
                
                if new_radius < best_new_radius - 1e-12:
                    best_new_radius = new_radius
                    best_flip = idx
            
            if best_flip is None:
                break
            
            signs[best_flip] = -signs[best_flip]
            current_radius, current_eigvec = compute_radius_and_eigvector(signs)
            
            if iteration % 10 == 0 and not is_connected_lift(signs):
                signs[best_flip] = -signs[best_flip]
                current_radius, current_eigvec = compute_radius_and_eigvector(signs)
                break
        
        if is_connected_lift(signs):
            return signs, current_radius
        else:
            return None, float('inf')
>>>>>>> REPLACE

And replace the fallback:

