import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 1. Define the graph of the 3x4 hexagonal cluster
    # Nodes 0..3 in row 0, 4..7 in row 1, 8..11 in row 2
    # Edges connect touching hexagons
    edges = []
    # Horizontal edges in each row
    for row in range(3):
        for col in range(3):
            i = row * 4 + col
            j = row * 4 + col + 1
            edges.append((i, j))
            
    # Vertical/Diagonal edges between rows
    # Row 0 (y=0) to Row 1 (y=sqrt(3)/2 * something)
    # In a hex grid, node (r,c) touches (r+1, c) and (r+1, c+1) if offset?
    # Let's assume a standard offset grid where row 1 is shifted right by 0.5 units.
    # Node (0,c) touches (1,c) and (1,c-1)?
    # Let's just hardcode the connections for a 3x4 block.
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    
    # Connections:
    # 0 touches 1, 4, 5
    # 1 touches 0, 2, 4, 5, 6
    # 2 touches 1, 3, 5, 6, 7
    # 3 touches 2, 6, 7
    
    # 4 touches 0, 1, 5, 8, 9
    # 5 touches 0, 1, 2, 4, 6, 8, 9, 10
    # 6 touches 1, 2, 3, 5, 7, 9, 10, 11
    # 7 touches 2, 3, 6, 10, 11
    
    # 8 touches 4, 5, 9
    # 9 touches 4, 5, 6, 8, 10
    # 10 touches 5, 6, 7, 9, 11
    # 11 touches 6, 7, 10

    # This is getting messy to hardcode. Let's generate based on lattice coords.
    # Lattice coords (q, r) for hex grid.
    # Let's map indices to axial coords.
    # Row 0: (0,0), (1,0), (2,0), (3,0)
    # Row 1: (0,1), (1,1), (2,1), (3,1) -> This is a parallelogram.
    # In a parallelogram of hexagons, each interior node has 6 neighbors.
    # Boundary nodes have fewer.
    
    # Let's just use the spectral embedding on the grid graph.
    # The graph is a 4x3 grid of nodes, but with hexagonal connectivity?
    # No, a 4x3 grid of *hexagons* forms a specific graph.
    # If we arrange them in a rectangular lattice of centers, the graph is just a grid graph (4x3).
    # But hexagons in a rectangular lattice of centers (spacing dx, dy) don't touch in the standard hex way unless dx=dy=sqrt(3) and offset.
    
    # Let's stick to the simplest interpretation:
    # The "spectral graph embedding" will recover the 2D coordinates of the nodes.
    # We define the adjacency matrix $A$ such that $A_{ij}=1$ if hexagons $i$ and $j$ are neighbors in the packing.
    # We use the packing topology of a 3x4 hexagonal block.
    
    # Topology of 3x4 hex block:
    # It is a subgraph of the infinite hexagonal lattice.
    # Let's generate the neighbors based on the lattice structure.
    # We'll place them in a virtual grid and check distances.
    
    # Initial guess positions for topology generation (not for final answer)
    # Row 0: y=0, x=0,1,2,3
    # Row 1: y=1, x=0.5, 1.5, 2.5, 3.5
    # Row 2: y=2, x=0,1,2,3
    # This is a "pointy-top" hex grid layout.
    
    pos_guess = []
    for r in range(3):
        for c in range(4):
            x = c + 0.5 * (r % 2)
            y = r * 0.866 # approx sqrt(3)/2
            pos_guess.append((x, y))
            
    # Determine edges based on proximity in this guess
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dx = pos_guess[i][0] - pos_guess[j][0]
            dy = pos_guess[i][1] - pos_guess[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.1: # Neighbors are ~1.0 apart in this scaled grid
                A[i, j] = 1
                A[j, i] = 1
                
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Compute eigenvectors
    # We want the 2 smallest non-zero eigenvalues.
    # L has one zero eigenvalue (constant vector).
    # The next two give the x and y coords.
    eigenvalues, eigenvectors = eigh(L)
    
    # Skip the first eigenvector (index 0) which is constant
    x_coords = eigenvectors[:, 1]
    y_coords = eigenvectors[:, 2]
    
    # Center the points
    x_coords -= np.mean(x_coords)
    y_coords -= np.mean(y_coords)
    
    # Normalize the scale
    # The distance between neighbors in the embedding should correspond to sqrt(3)
    # Calculate average distance between connected nodes
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            if A[i, j] == 1:
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dists.append(np.sqrt(dx*dx + dy*dy))
    avg_dist = np.mean(dists)
    scale = np.sqrt(3) / avg_dist
    
    x_coords *= scale
    y_coords *= scale
    
    # Initial positions
    inner_pos = np.column_stack((x_coords, y_coords))
    
    # Now optimize container
    # Variables: rotation of inner cluster (theta), rotation of container (phi), center offset (cx, cy)
    # Note: Translation of inner cluster relative to container is handled by cx, cy.
    # We can fix the inner cluster center at (0,0) and move the container.
    
    def get_container_side(theta, phi, cx, cy):
        # Rotate inner points by theta
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_pos = inner_pos @ R.T
        
        # Translate inner points? No, we move the container.
        # Container is centered at (cx, cy) with rotation phi.
        # We want to find min s such that all inner hexagons are inside.
        # Inner hexagons are at rotated_pos, with orientation 0 (aligned with x-axis? No, hexagons have fixed shape).
        # The task allows rotation of inner hexagons. In a lattice packing, they are parallel.
        # We assume inner hexagons are rotated by theta as well? 
        # "each inner hexagon may translate and rotate".
        # If we rotate the whole cluster, we should rotate the hexagons too to maintain the packing.
        # So inner hexagon orientation = theta.
        
        # Container normals
        # Container is a regular hexagon.
        # Normals are at angles phi, phi+60, ...
        normals = []
        for k in range(6):
            angle = phi + k * np.pi / 3
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # For each inner hexagon, we need to check if it is inside.
        # A hexagon of side 1 centered at p, rotated by theta.
        # Its vertices are p + R(theta) * v_k, where v_k are vertices of unit hex at origin.
        # Unit hex vertices (flat-top? pointy-top?):
        # Let's use standard: vertices at angles 0, 60, ...
        # v_k = [cos(k*pi/3), sin(k*pi/3)]
        
        v_hex = np.array([[np.cos(k*np.pi/3), np.sin(k*np.pi/3)] for k in range(6)])
        
        max_projection = 0
        for i in range(N):
            center = rotated_pos[i]
            # Vertices of this hexagon
            verts = center + (v_hex @ R.T)
            # For each container normal, check max projection
            # Projection of vertex v onto normal n is v . n
            # We need max over all vertices and all normals?
            # No, the condition is: for each normal n_k, max_{v in all_hex_verts} v . n_k <= a + c . n_k ?
            # Container is defined by (p - c) . n_k <= a
            # So p . n_k <= c . n_k + a
            # a = s * sqrt(3) / 2
            # We want to find min s.
            # s = 2/sqrt(3) * max_k ( max_p (p . n_k) - c . n_k )
            
            # To avoid iterating over all vertices for each normal (expensive),
            # we can calculate the support function of the inner set in direction n_k.
            # Support function h(n) = max_{p in Set} p . n
            # For a union of hexagons, h(n) = max_i h_i(n)
            # For a single hexagon centered at c_i, rotated by theta:
            # h_i(n) = c_i . n + h_unit_hex(n - theta)
            # h_unit_hex(alpha) is the support function of a unit hexagon at angle alpha.
            # For a regular hexagon, h(alpha) = cos(alpha mod 60 - 30) * (sqrt(3)/2) ?
            # Actually, the apothem is sqrt(3)/2. The support function is piecewise cosine.
            # h(alpha) = (sqrt(3)/2) / cos( (alpha % (pi/3)) - pi/6 )
            
            for k in range(6):
                n = normals[k]
                # Calculate support of all inner hexagons in direction n
                max_proj = 0
                for i in range(N):
                    c_i = rotated_pos[i]
                    proj_center = np.dot(c_i, n)
                    # Relative angle of n with respect to hex orientation
                    # Hex is rotated by theta.
                    # Normal n is at angle beta.
                    # Relative angle alpha = beta - theta
                    beta = np.arctan2(n[1], n[0])
                    alpha = beta - theta
                    # Normalize alpha to [-pi/6, pi/6] relative to the flat side?
                    # The hexagon has flats at angles 0, 60, ... if theta=0?
                    # My v_hex was angles 0, 60... so flats are at 30, 90...?
                    # Let's check v_hex: (1,0), (0.5, 0.866)...
                    # Edge from (1,0) to (0.5, 0.866). Midpoint (0.75, 0.433).
                    # Angle of midpoint: atan(0.433/0.75) = 30 deg.
                    # So flats are at 30, 90, 150...
                    # So support function peak is at 30 deg.
                    # h(alpha) = (sqrt(3)/2) / cos(alpha - pi/6)
                    # Valid for alpha in [0, pi/3]?
                    
                    # Simpler: Just compute vertices. N=12, 6 verts each, 6 normals.
                    # 12*6*6 = 432 dot products. Very fast.
                    pass
                
                # Let's just do the vertex check for clarity and correctness
                # It's only 432 ops.
                max_vertex_proj = -np.inf
                for i in range(N):
                    c_i = rotated_pos[i]
                    verts = c_i + (v_hex @ R.T)
                    projs = verts @ n
                    max_vertex_proj = max(max_vertex_proj, np.max(projs))
                
                # Container constraint: max_vertex_proj <= c_container . n + a
                # a = s * sqrt(3) / 2
                # s >= 2/sqrt(3) * (max_vertex_proj - c_container . n)
                
                c_cont = np.array([cx, cy])
                proj_c = np.dot(c_cont, n)
                required_s = (2.0 / np.sqrt(3)) * (max_vertex_proj - proj_c)
                if required_s > max_projection:
                    max_projection = required_s
                    
        return max_projection

    # Initial guess for optimization
    theta0 = 0.0
    phi0 = 0.0
    cx0 = 0.0
    cy0 = 0.0
    
    # Optimize
    # We want to minimize max_projection (which is s)
    # But s must be positive.
    
    def objective(params):
        theta, phi, cx, cy = params
        s = get_container_side(theta, phi, cx, cy)
        return s
    
    x0 = [theta0, phi0, cx0, cy0]
    
    # Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    
    theta_opt, phi_opt, cx_opt, cy_opt = res.x
    s_opt = res.fun
    
    # Construct final inner list
    c, s_rot = np.cos(theta_opt), np.sin(theta_opt)
    R = np.array([[c, -s_rot], [s_rot, c]])
    final_pos = inner_pos @ R.T
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(theta_opt)])
        
    # Container center is (cx_opt, cy_opt) relative to the rotated inner cluster?
    # In get_container_side, we used:
    # rotated_pos (inner)
    # c_container = (cx, cy)
    # So the container center is (cx_opt, cy_opt) in the coordinate system where inner cluster is at origin?
    # No, inner cluster was centered at origin, then rotated.
    # So container center is (cx_opt, cy_opt).
    
    return inner, (cx_opt, cy_opt), s_opt, np.degrees(phi_opt)

# EVOLVE-BLOCK-END
