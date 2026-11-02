def add_bcl_block(i0, j0, k0, start_col):
    """Add BCL rank-23 decomposition for 3x3x3 block at offsets (i0,j0,k0)."""
    # Strassen part for i∈{i0,i0+1}, j∈{j0,j0+1}, k∈{k0,k0+1}
    # ... (same as add_strassen_block but with range 3x3x3)
    
    # 16 additional terms
    # ...
