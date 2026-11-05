def _sort_key(name):
    try:
        return int(name[1:])
    except:
        return float('inf') # Or some large number, or just use name
