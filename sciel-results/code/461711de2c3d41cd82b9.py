def _enforce_symmetry(cell, size):
    sym = cell.copy()
    # Mirror x
    sym[:, size//2:] = sym[:, :size//2:-1] if size % 2 == 0 else sym[:, :size//2:-1] # Careful with center
    # Actually, simpler:
    sym = np.flipud(sym) # Wait, symmetry is about center.
    # cell[i][j] == cell[N-1-i][j] == cell[i][N-1-j] == cell[N-1-i][N-1-j]
    # This is equivalent to:
    # 1. Flip vertically
    # 2. Flip horizontally
    # If we just take the top-left quadrant and tile it, we enforce it.
    # But we need to ensure the input is already consistent or we just overwrite.
    # The function is used to *create* a symmetric cell from a potentially asymmetric one (by taking top-left and mirroring) or to *fix* a cell after a flip.
    # In the seed, `_params_to_cell` creates a symmetric field, then binarizes, then `_enforce_symmetry` is called. The binarization might break symmetry if the threshold crosses asymmetrically? No, the field is symmetric, so the binarized cell is symmetric.
    # In Phase 2, we flip a pixel in `cell`, which breaks symmetry, then call `_enforce_symmetry`.
    # So `_enforce_symmetry` should take the top-left quadrant (including the central lines if N is odd, but N=32 is even) and mirror it.
    
    # For N=32 (even):
    # Quadrant is 16x16.
    # Top-Left: cell[0:16, 0:16]
    # Top-Right: flip(Top-Left, axis=1)
    # Bottom-Left: flip(Top-Left, axis=0)
    # Bottom-Right: flip(Top-Left, axis=0, 1)
    
    h = size // 2
    tl = cell[:h, :h]
    tr = np.flipud(tl) # No, flip left-right for top right?
    # cell[i, j] for i < h, j >= h is cell[i, N-1-j].
    # So Top-Right is horizontal flip of Top-Left.
    tr = np.fliplr(tl)
    bl = np.flipud(tl)
    br = np.fliplr(np.flipud(tl))
    
    return np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
