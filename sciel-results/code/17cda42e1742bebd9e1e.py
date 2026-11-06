    # Store all outputs
    for v in topo_order:
        if v in outputs:
            ...
    
    # Limit to 20000 operations
    if len(operations) > 20000:
        operations = operations[:20000]
