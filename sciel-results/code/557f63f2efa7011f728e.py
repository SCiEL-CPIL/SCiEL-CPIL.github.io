# Instead of np.unique, use a small set
seen = set()
for v in window:
    seen.add(v)
distinct = len(seen)
