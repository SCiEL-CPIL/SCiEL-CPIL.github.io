    for idx in sup_order:
        if selected_mask[idx]:
            continue
        if not any(j in selected_mask for j in sup_adj[idx]):
            selected.append(idx)
            selected_mask[idx] = True
    