import itertools
import time

# The task is to find a packing of 6-subsets on 12 points such that no 5-subset is repeated.
# This is equivalent to finding an Exact Cover of the 792 5-subsets using the 924 available 6-subsets.
# Each 6-subset covers exactly 6 distinct 5-subsets.
# 132 blocks * 6 5-subsets/block = 792 5-subsets.
# We use Knuth's Algorithm X (Dancing Links) to solve this exact cover problem.

def construct():
    start_time = time.time()
    
    # 1. Enumerate all 5-subsets and assign them IDs
    # There are C(12, 5) = 792 five-subsets.
    # We map each 5-subset (tuple) to a unique integer ID.
    five_subsets = list(itertools.combinations(range(12), 5))
    five_to_id = {s: i for i, s in enumerate(five_subsets)}
    
    # 2. Enumerate all 6-subsets (candidate blocks)
    # There are C(12, 6) = 924 six-subsets.
    six_subsets = list(itertools.combinations(range(12), 6))
    
    # 3. Build the Exact Cover matrix
    # Columns: 792 (the 5-subsets)
    # Rows: 924 (the 6-subsets)
    # Each row has 6 1s, corresponding to the 5-subsets contained in that 6-subset.
    
    # We will build the data structure for Dancing Links.
    # L, R, U, D arrays for the circular doubly linked lists.
    # C (column) and R (row) pointers.
    
    N_cols = len(five_subsets)
    N_rows = len(six_subsets)
    
    # Initialize the header row
    # We use 0 as the header.
    # Nodes are indexed 1..TotalNodes
    # Total nodes = 1 (header) + N_cols (col headers) + sum(row_lengths)
    # sum(row_lengths) = 924 * 6 = 5544
    # Total nodes = 1 + 792 + 5544 = 6337
    
    total_nodes = 1 + N_cols + N_rows * 6
    # Arrays for L, R, U, D, C, R_idx, S (size of column)
    L = [0] * total_nodes
    R = [0] * total_nodes
    U = [0] * total_nodes
    D = [0] * total_nodes
    C = [0] * total_nodes
    R_idx = [0] * total_nodes  # Which row this node belongs to
    S = [0] * (N_cols + 1)    # Size of each column (header index 1..N_cols)
    
    # Initialize header row
    for j in range(total_nodes):
        L[j] = j - 1
        R[j] = j + 1
        U[j] = j
        D[j] = j
    L[0] = total_nodes - 1
    R[total_nodes - 1] = 0
    
    # Initialize column headers
    # Column j corresponds to 5-subset ID j-1.
    # Header nodes for columns are 1..N_cols
    for j in range(1, N_cols + 1):
        S[j] = 0
        # Link column headers into the header row
        L[j] = j - 1
        R[j] = j + 1
        # U[j] and D[j] remain j for now (empty column)
        
    L[1] = 0
    R[0] = 1
    R[N_cols] = total_nodes - 1
    L[total_nodes - 1] = N_cols
    
    # Fill in the rows
    # Node index starts at N_cols + 1
    node_idx = N_cols + 1
    for r, block in enumerate(six_subsets):
        # Find the 5-subsets contained in this 6-subset
        # Remove each element from the block to get a 5-subset
        for i in range(6):
            # Create 5-subset by removing block[i]
            five = tuple(x for x in block if x != block[i])
            col_id = five_to_id[five] + 1  # 1-based index for column header
            
            # Create node
            L[node_idx] = L[col_id]
            R[node_idx] = col_id
            R[L[col_id]] = node_idx
            L[col_id] = node_idx
            
            U[node_idx] = U[col_id]
            D[node_idx] = col_id
            D[U[col_id]] = node_idx
            U[col_id] = node_idx
            
            C[node_idx] = col_id
            R_idx[node_idx] = r
            S[col_id] += 1
            
            node_idx += 1
    
    # Ensure the circular links are correct for the last row
    # The last node added is node_idx - 1
    # It should link back to the first node of the last row? 
    # Actually, the row links are handled by L/R in the row direction, 
    # but in DLX, L/R are used for the column lists. 
    # The row links are implicit in the structure? 
    # No, standard DLX uses L/R for horizontal (row) and U/D for vertical (column).
    # Wait, the standard implementation:
    # L/R link nodes horizontally within a row.
    # U/D link nodes vertically within a column.
    # My initialization above linked L/R to column headers, which is wrong for row links.
    
    # Let's restart the DLX structure setup correctly.
    
    # Reset
    total_nodes = 1 + N_cols + N_rows * 6
    L = [0] * total_nodes
    R = [0] * total_nodes
    U = [0] * total_nodes
    D = [0] * total_nodes
    C = [0] * total_nodes
    R_idx = [0] * total_nodes
    S = [0] * (N_cols + 1)
    
    # Header row (node 0)
    for j in range(total_nodes):
        L[j] = j - 1
        R[j] = j + 1
        U[j] = j
        D[j] = j
    L[0] = total_nodes - 1
    R[total_nodes - 1] = 0
    
    # Column headers (nodes 1..N_cols)
    for j in range(1, N_cols + 1):
        S[j] = 0
        # Link column headers into the main header row
        # This part is tricky. The main header row contains the column headers.
        # The column headers themselves form a circular list.
        # Node 0 is the root.
        # R[0] = 1, L[1] = 0
        # R[1] = 2, L[2] = 1
        # ...
        # R[N_cols] = total_nodes - 1 (end of list? No, the column headers are in the main row)
        # Actually, the main row contains ONLY the column headers.
        # So R[0] = 1, L[1] = 0
        # R[1] = 2, L[2] = 1
        # ...
        # R[N_cols] = 0? No, it's a circular list.
        # R[N_cols] = 0 is not right if 0 is the root.
        # Standard DLX:
        # Header row: 0 <-> 1 <-> 2 <-> ... <-> N_cols <-> 0
        # So:
        R[0] = 1
        L[1] = 0
        for j in range(1, N_cols):
            R[j] = j + 1
            L[j + 1] = j
        R[N_cols] = 0
        L[0] = N_cols
        
        # U and D for column headers point to themselves initially (empty column)
        U[j] = j
        D[j] = j
        
    # Now add the data nodes (rows)
    # Node indices start from N_cols + 1
    node_idx = N_cols + 1
    for r, block in enumerate(six_subsets):
        # For each element in the row, create a node
        # We need to link them horizontally (L/R) within the row
        # and vertically (U/D) within the column.
        
        # First, find the columns for this row
        cols_in_row = []
        for i in range(6):
            five = tuple(x for x in block if x != block[i])
            col_id = five_to_id[five] + 1
            cols_in_row.append(col_id)
            
        # Create nodes and link horizontally
        first_node_in_row = node_idx
        for k, col_id in enumerate(cols_in_row):
            node = node_idx
            
            # Horizontal links (L/R)
            if k == 0:
                L[node] = node_idx + 5 # Last node in this row
                R[node] = node_idx + 1
            else:
                L[node] = node_idx + k - 1
                R[node] = node_idx + k + 1 if k < 5 else first_node_in_row
            
            # Vertical links (U/D) into the column
            C[node] = col_id
            R_idx[node] = r
            
            # Insert node at the bottom of the column list
            U[node] = U[col_id]
            D[node] = col_id
            D[U[col_id]] = node
            U[col_id] = node
            
            S[col_id] += 1
            
            node_idx += 1
        
        # Fix the last node's R pointer
        last_node = node_idx - 1
        R[last_node] = first_node_in_row
        L[first_node_in_row] = last_node

    # Algorithm X
    solution_rows = []
    
    def search(k):
        if time.time() - start_time > 140:
            return False # Timeout
            
        if R[0] == 0:
            # Solution found
            # Extract the solution
            # We need to trace back the choices.
            # It's easier to store the choice at each level.
            # Let's modify search to store the choice.
            return True
        
        # Choose column with minimum size
        j = R[0]
        min_size = S[j]
        while j != 0:
            if S[j] < min_size:
                min_size = S[j]
                j = R[j]
            j = R[j]
        
        # Cover column j
        R[L[j]] = R[j]
        L[R[j]] = L[j]
        i = D[j]
        while i != j:
            j2 = R[i]
            while j2 != i:
                U[D[j2]] = U[j2]
                D[U[j2]] = D[j2]
                S[C[j2]] -= 1
                j2 = R[j2]
            i = D[i]
        
        # Recurse
        i = D[j]
        found = False
        while i != j and not found:
            # Select row i
            solution_rows.append(R_idx[i])
            
            # Cover other columns in this row
            j2 = R[i]
            while j2 != i:
                # Cover column C[j2]
                col = C[j2]
                R[L[col]] = R[col]
                L[R[col]] = L[col]
                
                i2 = D[col]
                while i2 != col:
                    j3 = R[i2]
                    while j3 != i2:
                        U[D[j3]] = U[j3]
                        D[U[j3]] = D[j3]
                        S[C[j3]] -= 1
                        j3 = R[j3]
                    i2 = D[i2]
                j2 = R[j2]
            
            found = search(k + 1)
            
            if not found:
                # Uncover other columns in this row (reverse order)
                j2 = L[i]
                while j2 != i:
                    col = C[j2]
                    # Uncover column col
                    i2 = U[col]
                    while i2 != col:
                        j3 = L[i2]
                        while j3 != i2:
                            S[C[j3]] += 1
                            U[D[j3]] = j3
                            D[U[j3]] = j3
                            j3 = L[j3]
                        i2 = U[i2]
                    
                    R[L[col]] = col
                    L[R[col]] = col
                    j2 = L[j2]
                
                solution_rows.pop()
            
            i = D[i]
        
        # Uncover column j
        i = U[j]
        while i != j:
            j2 = L[i]
            while j2 != i:
                S[C[j2]] += 1
                U[D[j2]] = j2
                D[U[j2]] = j2
                j2 = L[j2]
            i = U[i]
        
        R[L[j]] = j
        L[R[j]] = j
        
        return found
    
    # Start search
    # Note: The solution_rows will contain the row indices (0..923)
    if search(0):
        # Convert row indices back to blocks
        blocks = [list(six_subsets[r]) for r in solution_rows]
        # Verify
        seen_5 = set()
        valid = True
        for b in blocks:
            for s in itertools.combinations(b, 5):
                if s in seen_5:
                    valid = False
                    break
                seen_5.add(s)
            if not valid:
                break
        if valid:
            return blocks
    
    # Fallback: Return empty or best effort if DLX fails (should not happen)
    return []

if __name__ == "__main__":
    blocks = construct()
    print(f"Found {len(blocks)} blocks")
