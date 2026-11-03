def _build_strassen_4x5_5x6():
    # ... existing code ...
    
    # Check if pruned 4x4x2 is better
    try:
        U2p, V2p, W2p = _build_strassen_4x4x2_pruned()
        rank2p = len(U2p[0])
    except Exception:
        rank2p = 999
        
    use_pruned = rank2p < 28
    
    if use_pruned:
        # Use pruned for part 2
        # ... mapping code for pruned ...
    else:
        # Use standard
        # ... mapping code for standard ...
