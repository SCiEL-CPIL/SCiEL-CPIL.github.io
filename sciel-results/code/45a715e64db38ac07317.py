valid = True
for u in adj[v]:
    if colors[u] == c:
        valid = False
        break
if not valid: continue

# Group neighbors by color
neighbors_by_color = {}
for u in adj[v]:
    cu = colors[u]
    if cu == -1: continue
    if cu == c: continue # Already handled
    if cu not in neighbors_by_color:
        neighbors_by_color[cu] = []
    neighbors_by_color[cu].append(u)

for cu, us in neighbors_by_color.items():
    p1, p2 = (c, cu) if c < cu else (cu, c)
    pi = pair_idx[(p1, p2)]
    # Check if any two in us are in the same component
    root_seen = set()
    for u in us:
        r = find(parents[pi], u)
        if r in root_seen:
            valid = False
            break
        root_seen.add(r)
    if not valid:
        break

if valid:
    # Proceed to update state
    # We need to union v with all u in neighbors_by_color
    # But v is new.
    # We can just add v to the component of the first u?
    # No, v connects to all of them.
    # If they are in distinct components, v merges them all into one.
    # So we union v with u1, then with u2, etc.
    # Since they were in distinct components, this is safe.
